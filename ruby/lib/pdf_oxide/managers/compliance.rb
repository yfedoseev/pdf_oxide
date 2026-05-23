# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for PDF compliance operations (PDF/A, PDF/X, PDF/UA)
    # Provides methods to validate and convert documents to compliant formats
    class Compliance < Base
      # PDF/A LEVELS
      PDF_A_1 = 1
      PDF_A_2 = 2
      PDF_A_3 = 3

      # PDF/X STANDARDS
      PDF_X_1 = 1
      PDF_X_3 = 3
      PDF_X_4 = 4

      # PDF/UA STANDARD
      PDF_UA_1 = 1

      # Validate PDF/A compliance
      # @param level [Integer] PDF/A level (1, 2, or 3)
      # @return [Hash] Validation results
      def validate_pdf_a(level = PDF_A_1)
        check_document!
        validate_compliance_level(level)

        result_handle = with_error_check('validate_pdf_a', level: level) do |error_ptr|
          FFI::Bindings.pdf_validate_pdf_a(@document.handle, level, error_ptr)
        end

        parse_compliance_result(result_handle, :pdf_a)
      end

      # Validate PDF/X compliance
      # @param standard [Integer] PDF/X standard (1, 3, or 4)
      # @return [Hash] Validation results
      def validate_pdf_x(standard = PDF_X_1)
        check_document!

        result_handle = with_error_check('validate_pdf_x', standard: standard) do |error_ptr|
          FFI::Bindings.pdf_validate_pdf_x(@document.handle, standard, error_ptr)
        end

        parse_pdf_x_result(result_handle, :pdf_x)
      end

      # Validate PDF/UA compliance
      # @return [Hash] Validation results
      def validate_pdf_ua
        check_document!

        result_handle = with_error_check('validate_pdf_ua') do |error_ptr|
          FFI::Bindings.pdf_validate_pdf_ua(@document.handle, PDF_UA_1, error_ptr)
        end

        parse_pdf_ua_result(result_handle, :pdf_ua)
      end

      # Check if document is PDF/A compliant
      # @param level [Integer] PDF/A level
      # @return [Boolean] Whether document is PDF/A compliant
      def is_pdf_a?(level = PDF_A_1)
        check_document!

        result = validate_pdf_a(level)
        result[:compliant] || false
      end

      # Check if document is PDF/X compliant
      # @param standard [Integer] PDF/X standard
      # @return [Boolean] Whether document is PDF/X compliant
      def is_pdf_x?(standard = PDF_X_1)
        check_document!

        result = validate_pdf_x(standard)
        result[:compliant] || false
      end

      # Check if document is PDF/UA compliant
      # @return [Boolean] Whether document is PDF/UA compliant
      def is_pdf_ua?
        check_document!

        result = validate_pdf_ua
        result[:compliant] || false
      end

      # Convert document to PDF/A
      # @param level [Integer] Target PDF/A level
      # @param output_path [String] Output file path
      # @return [Boolean] Whether conversion succeeded
      def convert_to_pdf_a(level, output_path)
        check_document!
        validate_compliance_level(level)
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('convert_to_pdf_a', level: level, path: output_path) do |error_ptr|
          FFI::Bindings.pdf_convert_to_pdf_a(@document.handle, level, error_ptr)
        end

        true
      end

      # Convert document to PDF/X
      # @param standard [Integer] Target PDF/X standard
      # @param output_path [String] Output file path
      # @return [Boolean] Whether conversion succeeded
      def convert_to_pdf_x(standard, output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('convert_to_pdf_x', standard: standard, path: output_path) do |error_ptr|
          FFI::Bindings.pdf_convert_to_pdf_x(@document.handle, standard, error_ptr)
        end

        true
      end

      # Convert document to PDF/UA
      # @param output_path [String] Output file path
      # @return [Boolean] Whether conversion succeeded
      def convert_to_pdf_ua(output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('convert_to_pdf_ua', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_convert_to_pdf_ua(@document.handle, PDF_UA_1, error_ptr)
        end

        true
      end

      # Get validation errors
      # @param compliance_type [Symbol] Type of compliance (:pdf_a, :pdf_x, :pdf_ua)
      # @return [Array<Hash>] Array of error information
      def get_validation_errors(compliance_type = :pdf_a)
        check_document!

        case compliance_type
        when :pdf_a
          result = validate_pdf_a
        when :pdf_x
          result = validate_pdf_x
        when :pdf_ua
          result = validate_pdf_ua
        else
          raise ::PdfOxide::ArgumentError, "Unknown compliance type: #{compliance_type}"
        end

        result[:errors] || []
      end

      # Get validation warnings
      # @param compliance_type [Symbol] Type of compliance
      # @return [Array<Hash>] Array of warning information
      def get_validation_warnings(compliance_type = :pdf_a)
        check_document!

        case compliance_type
        when :pdf_a
          result = validate_pdf_a
        when :pdf_x
          result = validate_pdf_x
        when :pdf_ua
          result = validate_pdf_ua
        else
          raise ::PdfOxide::ArgumentError, "Unknown compliance type: #{compliance_type}"
        end

        result[:warnings] || []
      end

      # Get compliance information
      # @return [Hash] Compliance information
      def compliance_info
        check_document!
        {
          pdf_a: { compliant: is_pdf_a?, level: PDF_A_1 },
          pdf_x: { compliant: is_pdf_x?, standard: PDF_X_1 },
          pdf_ua: { compliant: is_pdf_ua? }
        }
      end

      # Validate against all standards (PDF/A, PDF/X, PDF/UA)
      # @return [Hash] Combined validation results for all standards
      # Validate document against all compliance standards at once
      #
      # Performs comprehensive validation of the document against all three
      # PDF compliance standards (PDF/A, PDF/X, and PDF/UA) in a single call.
      # Returns detailed results for each standard plus a summary.
      #
      # This is more efficient than calling validate_pdf_a, validate_pdf_x,
      # and validate_pdf_ua separately.
      #
      # @return [Hash] Comprehensive validation results including:
      #   - :timestamp [Integer] Unix timestamp of validation
      #   - :pdf_a [Hash] PDF/A validation results
      #   - :pdf_x [Hash] PDF/X validation results
      #   - :pdf_ua [Hash] PDF/UA validation results
      #   - :summary [Hash] Quick reference with compliance status and issue count
      # @raise [PdfException] if document is invalid or closed
      # @example
      #   results = manager.validate_all_standards
      #   if results[:summary][:pdf_a_compliant]
      #     puts "Document is PDF/A compliant"
      #   end
      #   puts "Total issues: #{results[:summary][:total_issues]}"
      def validate_all_standards
        check_document!

        {
          timestamp: Time.now.to_i,
          pdf_a: validate_pdf_a(PDF_A_1),
          pdf_x: validate_pdf_x(PDF_X_1),
          pdf_ua: validate_pdf_ua,
          summary: {
            pdf_a_compliant: is_pdf_a?,
            pdf_x_compliant: is_pdf_x?,
            pdf_ua_compliant: is_pdf_ua?,
            total_issues: (get_validation_errors(:pdf_a).count +
                          get_validation_errors(:pdf_x).count +
                          get_validation_errors(:pdf_ua).count)
          }
        }
      end

      # Get compliance recommendations
      #
      # Analyzes the document's compliance status and returns actionable
      # recommendations. Recommendations are specific to which standards are
      # failing and what actions would improve compliance.
      #
      # The method provides:
      # - Specific issues found for non-compliant standards
      # - Recommendations for remediation
      # - Positive feedback when all standards are met
      #
      # @return [Array<String>] Array of human-readable recommendations
      # @raise [PdfException] if document is invalid or closed
      # @example
      #   recommendations = manager.get_compliance_recommendations
      #   recommendations.each do |rec|
      #     puts "→ #{rec}"
      #   end
      #   # Output:
      #   # → Document is not PDF/A compliant. Issues found: 3
      #   # → Consider converting to PDF/A-1 for archival purposes
      #   # → Document is not PDF/UA compliant (not accessible). Issues found: 2
      def get_compliance_recommendations
        check_document!

        recommendations = []

        # Check PDF/A compliance
        unless is_pdf_a?
          errors = get_validation_errors(:pdf_a)
          if errors.any?
            recommendations << "Document is not PDF/A compliant. Issues found: #{errors.count}"
            recommendations << "Consider converting to PDF/A-1 for archival purposes"
          end
        end

        # Check PDF/X compliance
        unless is_pdf_x?
          errors = get_validation_errors(:pdf_x)
          if errors.any?
            recommendations << "Document is not PDF/X compliant. Issues found: #{errors.count}"
            recommendations << "Consider converting to PDF/X-1 for print workflows"
          end
        end

        # Check PDF/UA compliance
        unless is_pdf_ua?
          errors = get_validation_errors(:pdf_ua)
          if errors.any?
            recommendations << "Document is not PDF/UA compliant (not accessible). Issues found: #{errors.count}"
            recommendations << "Add structure tags and alt text for accessibility"
          end
        end

        # If no issues, add positive recommendations
        if recommendations.empty?
          recommendations << "Document is compliant with all tested standards"
          recommendations << "Continue to validate periodically for compliance"
        end

        recommendations
      end

      # Convert PDF/A level to string representation
      #
      # Converts a PDF/A level constant to its human-readable string form.
      # Useful for displaying compliance levels in logs and UI.
      #
      # PDF/A levels:
      # - PDF/A-1: Basic archival format (2005)
      # - PDF/A-2: Enhanced archival with improved compression (2011)
      # - PDF/A-3: With support for embedded files (2012)
      #
      # @param level [Integer] PDF/A level constant (PDF_A_1, PDF_A_2, PDF_A_3)
      # @return [String] String representation (e.g., "PDF/A-1", "PDF/A-2")
      # @example
      #   str = manager.pdf_a_level_to_string(Compliance::PDF_A_2)
      #   puts str  # => "PDF/A-2"
      def pdf_a_level_to_string(level = PDF_A_1)
        case level
        when PDF_A_1
          'PDF/A-1'
        when PDF_A_2
          'PDF/A-2'
        when PDF_A_3
          'PDF/A-3'
        else
          'PDF/A-Unknown'
        end
      end

      # Convert PDF/X standard to string representation
      #
      # Converts a PDF/X standard constant to its human-readable string form.
      # Useful for displaying print compliance levels in reports and logs.
      #
      # PDF/X standards:
      # - PDF/X-1: Basic exchange format for print (2001)
      # - PDF/X-3: With color management (2002)
      # - PDF/X-4: Latest standard with enhanced color management (2008)
      #
      # @param standard [Integer] PDF/X standard constant (PDF_X_1, PDF_X_3, PDF_X_4)
      # @return [String] String representation (e.g., "PDF/X-1", "PDF/X-3", "PDF/X-4")
      # @example
      #   str = manager.pdf_x_standard_to_string(Compliance::PDF_X_3)
      #   puts str  # => "PDF/X-3"
      def pdf_x_standard_to_string(standard = PDF_X_1)
        case standard
        when PDF_X_1
          'PDF/X-1'
        when PDF_X_3
          'PDF/X-3'
        when PDF_X_4
          'PDF/X-4'
        else
          'PDF/X-Unknown'
        end
      end

      # Convert PDF/UA to string representation
      # @return [String] String representation
      def pdf_ua_to_string
        'PDF/UA-1'
      end

      private

      def validate_compliance_level(level)
        raise ::PdfOxide::ArgumentError, "Invalid PDF/A level: #{level}" unless [PDF_A_1, PDF_A_2, PDF_A_3].include?(level)
      end

      def parse_compliance_result(handle, type)
        return { compliant: false, errors: [], warnings: [] } if handle.nil? || handle.null?

        begin
          {
            type: type,
            compliant: FFI::Bindings.pdf_pdf_a_is_compliant(handle),
            error_count: FFI::Bindings.pdf_pdf_a_error_count(handle),
            warning_count: FFI::Bindings.pdf_pdf_a_warning_count(handle),
            report: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_pdf_a_get_report(handle, ::FFI::MemoryPointer.new(:int32))
            ) || '',
            errors: [],
            warnings: []
          }
        ensure
          FFI::Bindings.pdf_pdf_a_results_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_pdf_x_result(handle, type)
        return { compliant: false, errors: [], warnings: [] } if handle.nil? || handle.null?

        begin
          {
            type: type,
            compliant: FFI::Bindings.pdf_pdf_x_is_compliant(handle),
            error_count: FFI::Bindings.pdf_pdf_x_error_count(handle),
            warning_count: FFI::Bindings.pdf_pdf_x_warning_count(handle),
            report: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_pdf_x_get_report(handle, ::FFI::MemoryPointer.new(:int32))
            ) || '',
            errors: [],
            warnings: []
          }
        ensure
          FFI::Bindings.pdf_pdf_x_results_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_pdf_ua_result(handle, type)
        return { compliant: false, errors: [] } if handle.nil? || handle.null?

        begin
          {
            type: type,
            compliant: FFI::Bindings.pdf_pdf_ua_is_accessible(handle),
            error_count: FFI::Bindings.pdf_pdf_ua_error_count(handle),
            errors: []
          }
        ensure
          FFI::Bindings.pdf_pdf_ua_results_free(handle) unless handle.nil? || handle.null?
        end
      end
    end
  end
end
