# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Meta-manager covering miscellaneous document-level operations that
    # do not belong to a more specific manager (metadata snapshotting,
    # statistics, incremental save, validate-all-standards, etc.).
    #
    # Named `MetaManager` to avoid collision with the user-facing
    # {PdfOxide::Document} class.
    class MetaManager < Base
      # Get document version
      # @return [String] PDF version (e.g., "1.4", "2.0")
      def get_version
        check_document!

        version_ptr = with_error_check('get_version') do |error_ptr|
          FFI::Bindings.pdf_document_get_version(@document.handle, error_ptr)
        end

        FFI::StringMarshaller.from_c_string(version_ptr) || '1.4'
      end

      # Get document metadata as raw string
      # @return [String] Raw metadata
      def get_metadata_raw
        check_document!

        metadata_ptr = with_error_check('get_metadata_raw') do |error_ptr|
          FFI::Bindings.pdf_document_get_metadata(@document.handle, error_ptr)
        end

        FFI::StringMarshaller.from_c_string(metadata_ptr) || ''
      end

      # Get file size in bytes
      # @return [Integer] File size
      def get_file_size
        check_document!

        with_error_check('get_file_size') do |error_ptr|
          FFI::Bindings.pdf_document_get_file_size(@document.handle, error_ptr)
        end
      end

      # Check if document has JavaScript
      # @return [Boolean] Whether document contains JavaScript
      def has_javascript?
        check_document!

        with_error_check('has_javascript') do |error_ptr|
          FFI::Bindings.pdf_document_has_javascript(@document.handle, error_ptr)
        end
      end

      # Check if document has structure tree (tagged PDF)
      # @return [Boolean] Whether document is tagged
      def has_structure_tree?
        check_document!

        with_error_check('has_structure_tree') do |error_ptr|
          FFI::Bindings.pdf_document_has_structure_tree(@document.handle, error_ptr)
        end
      end

      # Check if document has valid signatures
      # @return [Boolean] Whether all signatures are valid
      def has_valid_signatures?
        check_document!

        with_error_check('has_valid_signatures') do |error_ptr|
          FFI::Bindings.pdf_document_has_valid_signatures(@document.handle, error_ptr)
        end
      end

      # Get page count
      # @return [Integer] Number of pages
      def get_page_count
        check_document!

        with_error_check('get_page_count') do |error_ptr|
          FFI::Bindings.pdf_document_get_page_count(@document.handle, error_ptr)
        end
      end

      # Check if document is encrypted
      # @return [Boolean] Whether document is encrypted
      def is_encrypted?
        check_document!

        with_error_check('is_encrypted') do |error_ptr|
          FFI::Bindings.pdf_document_is_encrypted(@document.handle, error_ptr)
        end
      end

      # Check if document requires password
      # @return [Boolean] Whether document needs password
      def requires_password?
        check_document!

        with_error_check('requires_password') do |error_ptr|
          FFI::Bindings.pdf_document_requires_password(@document.handle, error_ptr)
        end
      end

      # Unlock document with password
      # @param password [String] Document password
      # @return [Boolean] Whether unlock succeeded
      def unlock_with_password(password)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Password cannot be empty' if password.nil? || password.empty?

        password_utf8 = FFI::StringMarshaller.to_utf8(password)

        with_error_check('unlock_with_password') do |error_ptr|
          FFI::Bindings.pdf_document_unlock_with_password(@document.handle, password_utf8, error_ptr)
        end
      end

      # Save document
      # @param output_path [String] File path to save to
      # @return [Boolean] Whether save succeeded
      def save(output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('save', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_document_save(@document.handle, output_path_utf8, error_ptr)
        end
      end

      # Save document incrementally (append changes)
      # @param output_path [String] File path to save to
      # @return [Boolean] Whether save succeeded
      def save_incremental(output_path)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('save_incremental', path: output_path) do |error_ptr|
          FFI::Bindings.pdf_document_save_incremental(@document.handle, output_path_utf8, error_ptr)
        end
      end

      # Validate document as PDF/A
      # @return [Boolean] Whether document is PDF/A compliant
      def validate_pdf_a
        check_document!

        with_error_check('validate_pdf_a') do |error_ptr|
          FFI::Bindings.pdf_document_validate_pdf_a(@document.handle, error_ptr)
        end
      end

      # Validate document as PDF/X
      # @return [Boolean] Whether document is PDF/X compliant
      def validate_pdf_x
        check_document!

        with_error_check('validate_pdf_x') do |error_ptr|
          FFI::Bindings.pdf_document_validate_pdf_x(@document.handle, error_ptr)
        end
      end

      # Validate document as PDF/UA
      # @return [Boolean] Whether document is PDF/UA compliant
      def validate_pdf_ua
        check_document!

        with_error_check('validate_pdf_ua') do |error_ptr|
          FFI::Bindings.pdf_document_validate_pdf_ua(@document.handle, error_ptr)
        end
      end

      # Search in page range
      # @param start_page [Integer] Start page (0-indexed)
      # @param end_page [Integer] End page (0-indexed)
      # @param query [String] Search query
      # @return [Array<Hash>] Search results
      def search_in_range(start_page, end_page, query)
        check_document!
        validate_page_index!(start_page)
        validate_page_index!(end_page)
        raise ::PdfOxide::ArgumentError, 'Query cannot be empty' if query.nil? || query.empty?

        query_utf8 = FFI::StringMarshaller.to_utf8(query)

        results_handle = with_error_check('search_in_range', start: start_page, end: end_page, query: query) do |error_ptr|
          FFI::Bindings.pdf_document_search_in_range(@document.handle, start_page, end_page, query_utf8, error_ptr)
        end

        parse_search_results(results_handle)
      end

      # Sign document
      # @param certificate_path [String] Path to certificate
      # @param output_path [String] Output file path
      # @param options [Hash] Signing options
      # @return [Boolean] Whether signing succeeded
      def sign(certificate_path, output_path, options = {})
        check_document!
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if certificate_path.nil? || certificate_path.empty?
        raise ::PdfOxide::ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

        cert_path_utf8 = FFI::StringMarshaller.to_utf8(certificate_path)
        output_path_utf8 = FFI::StringMarshaller.to_utf8(output_path)

        with_error_check('sign', cert: certificate_path, output: output_path) do |error_ptr|
          FFI::Bindings.pdf_document_sign(
            @document.handle,
            cert_path_utf8,
            output_path_utf8,
            error_ptr
          )
        end
      end

      # Estimate document processing time
      # @return [Float] Estimated time in milliseconds
      def estimate_processing_time
        check_document!

        with_error_check('estimate_processing_time') do |error_ptr|
          FFI::Bindings.pdf_estimate_processing_time(@document.handle, error_ptr)
        end
      end

      # Perform OCR on entire document
      # @param language [String] Language code (e.g., 'en', 'fr')
      # @return [Hash] OCR results
      def ocr_document(language = 'en')
        check_document!

        lang_utf8 = FFI::StringMarshaller.to_utf8(language)

        ocr_result_ptr = with_error_check('ocr_document', language: language) do |error_ptr|
          FFI::Bindings.pdf_document_ocr_page(@document.handle, 0, lang_utf8, error_ptr)
        end

        parse_ocr_result(ocr_result_ptr)
      end

      # Check if ML model is available
      # @return [Boolean] Whether ML model is available
      def ml_model_available?
        with_error_check('ml_model_available') do |error_ptr|
          FFI::Bindings.pdf_ml_model_available(error_ptr)
        end
      end

      # Get ML status
      # @return [String] ML model status
      def ml_get_status
        status_ptr = with_error_check('ml_get_status') do |error_ptr|
          FFI::Bindings.pdf_ml_get_status(error_ptr)
        end

        FFI::StringMarshaller.from_c_string(status_ptr) || 'unavailable'
      end

      # Get validation errors for PDF/A
      # @return [Array<String>] List of errors
      def get_pdf_a_errors
        return [] unless validate_pdf_a

        errors = []
        error_ptr = with_error_check('get_pdf_a_errors') do |error_ptr|
          FFI::Bindings.pdf_pdf_a_get_error(error_ptr)
        end

        errors << FFI::StringMarshaller.from_c_string(error_ptr) if error_ptr
        errors
      end

      # Get validation warnings for PDF/A
      # @return [Array<String>] List of warnings
      def get_pdf_a_warnings
        return [] unless validate_pdf_a

        warnings = []
        warning_ptr = with_error_check('get_pdf_a_warnings') do |error_ptr|
          FFI::Bindings.pdf_pdf_a_get_warning(error_ptr)
        end

        warnings << FFI::StringMarshaller.from_c_string(warning_ptr) if warning_ptr
        warnings
      end

      # Get validation errors for PDF/X
      # @return [Array<String>] List of errors
      def get_pdf_x_errors
        return [] unless validate_pdf_x

        errors = []
        error_ptr = with_error_check('get_pdf_x_errors') do |error_ptr|
          FFI::Bindings.pdf_pdf_x_get_error(error_ptr)
        end

        errors << FFI::StringMarshaller.from_c_string(error_ptr) if error_ptr
        errors
      end

      # Get validation warnings for PDF/X
      # @return [Array<String>] List of warnings
      def get_pdf_x_warnings
        return [] unless validate_pdf_x

        warnings = []
        warning_ptr = with_error_check('get_pdf_x_warnings') do |error_ptr|
          FFI::Bindings.pdf_pdf_x_get_warning(error_ptr)
        end

        warnings << FFI::StringMarshaller.from_c_string(warning_ptr) if warning_ptr
        warnings
      end

      # Get validation errors for PDF/UA
      # @return [Array<String>] List of errors
      def get_pdf_ua_errors
        return [] unless validate_pdf_ua

        errors = []
        error_ptr = with_error_check('get_pdf_ua_errors') do |error_ptr|
          FFI::Bindings.pdf_pdf_ua_get_error(error_ptr)
        end

        errors << FFI::StringMarshaller.from_c_string(error_ptr) if error_ptr
        errors
      end

      # Document statistics
      # @return [Hash] Complete document information
      def statistics
        {
          version: get_version,
          page_count: get_page_count,
          file_size: get_file_size,
          is_encrypted: is_encrypted?,
          requires_password: requires_password?,
          has_javascript: has_javascript?,
          has_structure_tree: has_structure_tree?,
          has_valid_signatures: has_valid_signatures?,
          is_pdf_a: validate_pdf_a,
          is_pdf_x: validate_pdf_x,
          is_pdf_ua: validate_pdf_ua,
          processing_time_ms: estimate_processing_time,
          timestamp: Time.now.to_i
        }
      end

      private

      def parse_search_results(handle)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_search_result_count(handle)

          results = count.times.map do |i|
            page = FFI::Bindings.pdf_oxide_search_result_get_page(handle, i)
            text_ptr = FFI::Bindings.pdf_oxide_search_result_get_text(handle, i)

            {
              page: page,
              text: FFI::StringMarshaller.read_c_string(text_ptr) || ''
            }
          end

          results
        ensure
          FFI::Bindings.pdf_oxide_search_result_free(handle)
        end
      end

      def parse_ocr_result(result_ptr)
        return {} if result_ptr.nil? || result_ptr.null?

        {
          text: FFI::StringMarshaller.from_c_string(
            FFI::Bindings.pdf_oxide_ocr_result_get_text(result_ptr)
          ) || '',
          confidence: FFI::Bindings.pdf_oxide_ocr_result_confidence(result_ptr)
        }
      end
    end
  end
end
