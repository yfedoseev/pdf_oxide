# frozen_string_literal: true

require 'json'
require_relative 'base'

module PdfOxide
  module Managers
    # Manager for digital signature operations
    # Provides methods to verify, add, and manage digital signatures on PDFs
    class Signature < Base
      # SIGNATURE STATUS CONSTANTS
      SIGNATURE_STATUS_UNKNOWN = 0
      SIGNATURE_STATUS_VALID = 1
      SIGNATURE_STATUS_INVALID = 2
      SIGNATURE_STATUS_UNTRUSTED = 3
      SIGNATURE_STATUS_SELF_SIGNED = 4

      SIGNATURE_STATUSES = {
        unknown: SIGNATURE_STATUS_UNKNOWN,
        valid: SIGNATURE_STATUS_VALID,
        invalid: SIGNATURE_STATUS_INVALID,
        untrusted: SIGNATURE_STATUS_UNTRUSTED,
        self_signed: SIGNATURE_STATUS_SELF_SIGNED
      }.freeze

      STATUS_NAMES = SIGNATURE_STATUSES.invert.freeze

      # Get count of signatures in document
      # @return [Integer] Number of signatures
      def signature_count
        check_document!
        with_error_check('signature_count') do |error_ptr|
          FFI::Bindings.pdf_document_get_signature_count(@document.handle, error_ptr)
        end
      end

      # Check if document has signatures
      # @return [Boolean] Whether document has signatures
      def has_signatures?
        check_document!
        signature_count > 0
      end

      # Get signature information
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Hash] Signature information
      def get_signature(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        sig_handle = with_error_check('get_signature', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature(@document.handle, signature_index, error_ptr)
        end

        parse_signature(sig_handle, signature_index)
      end

      # Verify a signature
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Hash] Verification results
      def verify_signature(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        result_handle = with_error_check('verify_signature', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_verify_signature(@document.handle, signature_index, error_ptr)
        end

        parse_verification_result(result_handle)
      end

      # Get signature signer name
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [String] Signer name
      def get_signer(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        FFI::StringMarshaller.from_c_string(
          with_error_check('get_signer', index: signature_index) do |error_ptr|
            FFI::Bindings.pdf_document_get_signature_signer(@document.handle, signature_index, error_ptr)
          end
        ) || 'Unknown'
      end

      # Get signature timestamp
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Integer, nil] Unix timestamp or nil if no timestamp
      def get_timestamp(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        with_error_check('get_timestamp', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature_timestamp(@document.handle, signature_index, error_ptr)
        end
      end

      # Get signature status
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Symbol] Signature status (:valid, :invalid, :untrusted, :self_signed, :unknown)
      def get_signature_status(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        status_int = with_error_check('get_signature_status', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature_status(@document.handle, signature_index, error_ptr)
        end

        STATUS_NAMES[status_int] || :unknown
      end

      # Check if signature is valid
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Boolean] Whether signature is valid
      def is_signature_valid?(signature_index)
        get_signature_status(signature_index) == :valid
      end

      # Get all signatures
      # @return [Array<Hash>] All signatures
      def get_all_signatures
        check_document!
        count = signature_count
        (0...count).map { |i| get_signature(i) }
      end

      # List all signatures with basic info
      # @return [Array<Hash>] List of signatures with signer and status
      def list_signatures
        check_document!
        (0...signature_count).map do |i|
          {
            index: i,
            signer: get_signer(i),
            status: get_signature_status(i),
            timestamp: get_timestamp(i)
          }
        end
      end

      # Add a signature to the document
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width of signature area
      # @param height [Float] Height of signature area
      # @param options [Hash] Additional options (certificate_path, password, etc.)
      # @return [Boolean] Whether operation succeeded
      def add_signature(page_index, x, y, width, height, options = {})
        check_document!
        validate_page_index!(page_index)

        certificate_path = options.fetch(:certificate_path, '')
        certificate_path_utf8 = FFI::StringMarshaller.to_utf8(certificate_path)

        with_error_check('add_signature', page: page_index, rect: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_add_signature(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            certificate_path_utf8,
            error_ptr
          )
        end
        true
      end

      # Get signature statistics
      # @return [Hash] Signature statistics
      def signature_statistics
        check_document!
        signatures = list_signatures

        valid_count = signatures.count { |sig| sig[:status] == :valid }
        invalid_count = signatures.count { |sig| sig[:status] == :invalid }

        {
          total_signatures: signatures.count,
          valid_signatures: valid_count,
          invalid_signatures: invalid_count,
          statuses: signatures.map { |sig| sig[:status] }.tally
        }
      end

      # Sign the document with visual appearance
      # @param page_index [Integer] Page index (0-indexed)
      # @param position [Hash] Signature position (:x, :y, :width, :height)
      # @param credentials [Types::SigningCredentials] Signing credentials
      # @param options [Hash] Additional signing options (reason, location, contact, algorithm)
      # @return [Boolean] Whether operation succeeded
      # @raise [PdfException] If signing fails
      def sign_with_appearance(page_index, position, credentials, options = {})
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Credentials must be a SigningCredentials object' \
          unless credentials.is_a?(Types::SigningCredentials)
        raise ::PdfOxide::ArgumentError, 'Position must be a hash with :x, :y, :width, :height' \
          unless position.is_a?(Hash) && %i[x y width height].all? { |k| position.key?(k) }

        reason = options.fetch(:reason, '')
        location = options.fetch(:location, '')
        contact = options.fetch(:contact, '')
        algorithm = options.fetch(:algorithm, 1) # SHA-256 default

        reason_utf8 = FFI::StringMarshaller.to_utf8(reason)
        location_utf8 = FFI::StringMarshaller.to_utf8(location)
        contact_utf8 = FFI::StringMarshaller.to_utf8(contact)

        with_error_check('sign_with_appearance', page: page_index, position: position) do |error_ptr|
          FFI::Bindings.pdf_document_sign_with_appearance(
            @document.handle,
            credentials.handle,
            page_index,
            position[:x].to_f,
            position[:y].to_f,
            position[:width].to_f,
            position[:height].to_f,
            reason_utf8,
            location_utf8,
            contact_utf8,
            algorithm,
            error_ptr
          )
        end

        true
      end

      # Add RFC 3161 timestamp to a signature
      # @param signature_index [Integer] Index of signature to timestamp
      # @param tsa_url [String] URL of Time Stamp Authority
      # @return [Boolean] Whether timestamp was added successfully
      # @raise [PdfException] If timestamp fails
      def add_timestamp(signature_index, tsa_url)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count
        raise ::PdfOxide::ArgumentError, 'TSA URL must be a string' unless tsa_url.is_a?(String)

        tsa_utf8 = FFI::StringMarshaller.to_utf8(tsa_url)

        with_error_check('add_timestamp', signature: signature_index, tsa_url: tsa_url) do |error_ptr|
          FFI::Bindings.pdf_add_timestamp(@document.handle, signature_index, tsa_utf8, error_ptr)
        end

        true
      end

      # Add a co-signature to the document
      # Adds an additional signature without removing previous ones
      # @param credentials [Types::SigningCredentials] Signing credentials
      # @param reason [String] Reason for signing (optional)
      # @return [Integer] Index of new signature
      # @raise [PdfException] If co-signing fails
      def co_sign(credentials, reason = '')
        check_document!
        raise ::PdfOxide::ArgumentError, 'Credentials must be a SigningCredentials object' \
          unless credentials.is_a?(Types::SigningCredentials)
        raise ::PdfOxide::ArgumentError, 'Reason must be a string' unless reason.is_a?(String)

        reason_utf8 = FFI::StringMarshaller.to_utf8(reason)

        with_error_check('co_sign', reason: reason) do |error_ptr|
          FFI::Bindings.pdf_document_co_sign(@document.handle, credentials.handle, reason_utf8, error_ptr)
        end

        signature_count - 1 # Return index of newly added signature
      end

      # Verify all signatures in the document
      # @param trusted_certs [Array<String>] Paths to trusted certificate files (optional)
      # @return [Array<Hash>] Verification results for each signature
      # @raise [PdfException] If verification fails
      def verify_all_signatures(trusted_certs = [])
        check_document!
        raise ::PdfOxide::ArgumentError, 'Trusted certs must be an array' unless trusted_certs.is_a?(Array)

        # Create pointer array for trusted certificates
        cert_ptrs = if trusted_certs.empty?
                      ::FFI::MemoryPointer.new(:pointer, 0)
                    else
                      cert_paths = trusted_certs.map { |p| FFI::StringMarshaller.to_utf8(p) }
                      ::FFI::MemoryPointer.new(:pointer, cert_paths.length)
                    end

        error_ptr = ::FFI::MemoryPointer.new(:pointer)
        results_handle = FFI::Bindings.pdf_verify_all_signatures(
          @document.handle,
          cert_ptrs,
          trusted_certs.length,
          error_ptr,
          error_ptr
        )

        return [] if results_handle.nil? || results_handle.null?

        # Parse verification results for each signature
        (0...signature_count).map do |i|
          {
            signature_index: i,
            valid: is_signature_valid?(i),
            status: get_signature_status(i),
            signer: get_signer(i),
            timestamp: get_timestamp(i)
          }
        end
      end

      # Check if signature has a timestamp
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Boolean] Whether signature has a timestamp
      def has_timestamp?(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        sig_handle = with_error_check('get_signature', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature(@document.handle, signature_index, error_ptr)
        end

        return false if sig_handle.nil? || sig_handle.null?

        begin
          FFI::Bindings.pdf_signature_has_timestamp(sig_handle)
        ensure
          FFI::Bindings.pdf_oxide_signature_free(sig_handle) unless sig_handle.nil? || sig_handle.null?
        end
      end

      # Get signature digest algorithm
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Symbol] Digest algorithm (:sha1, :sha256, :sha384, :sha512, :unknown)
      def get_signature_algorithm(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        sig_handle = with_error_check('get_signature', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature(@document.handle, signature_index, error_ptr)
        end

        return :unknown if sig_handle.nil? || sig_handle.null?

        begin
          algorithm_int = FFI::Bindings.pdf_signature_get_digest_algorithm(sig_handle)
          DIGEST_ALGORITHM_NAMES[algorithm_int] || :unknown
        ensure
          FFI::Bindings.pdf_oxide_signature_free(sig_handle) unless sig_handle.nil? || sig_handle.null?
        end
      end

      # Export signature as JSON
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Hash] Signature as JSON-compatible hash
      def export_signature_json(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        sig_handle = with_error_check('get_signature', index: signature_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_signature(@document.handle, signature_index, error_ptr)
        end

        return {} if sig_handle.nil? || sig_handle.null?

        begin
          error_ptr = ::FFI::MemoryPointer.new(:pointer)
          json_str = FFI::Bindings.pdf_signature_to_json(sig_handle, error_ptr)

          return JSON.parse(FFI::StringMarshaller.from_c_string(json_str) || '{}')
        rescue => e
          { error: e.message }
        ensure
          FFI::Bindings.pdf_oxide_signature_free(sig_handle) unless sig_handle.nil? || sig_handle.null?
        end
      end

      # Remove a signature from the document
      # @param signature_index [Integer] Index of signature (0-indexed)
      # @return [Boolean] Whether signature was removed
      def remove_signature(signature_index)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Signature index must be >= 0' if signature_index < 0
        raise ::PdfOxide::ArgumentError, "Signature index #{signature_index} out of range" if signature_index >= signature_count

        with_error_check('remove_signature', signature: signature_index) do |error_ptr|
          FFI::Bindings.pdf_remove_signature(@document.handle, signature_index, error_ptr)
        end

        true
      end

      # Clear all signatures from the document
      # @return [Integer] Number of signatures removed
      def clear_all_signatures
        check_document!
        count = signature_count

        with_error_check('clear_all_signatures') do |error_ptr|
          FFI::Bindings.pdf_clear_all_signatures(@document.handle, error_ptr)
        end

        count
      end

      # DIGEST ALGORITHM CONSTANTS
      DIGEST_ALGORITHM_SHA1 = 0
      DIGEST_ALGORITHM_SHA256 = 1
      DIGEST_ALGORITHM_SHA384 = 2
      DIGEST_ALGORITHM_SHA512 = 3

      DIGEST_ALGORITHM_NAMES = {
        DIGEST_ALGORITHM_SHA1 => :sha1,
        DIGEST_ALGORITHM_SHA256 => :sha256,
        DIGEST_ALGORITHM_SHA384 => :sha384,
        DIGEST_ALGORITHM_SHA512 => :sha512
      }.freeze

      private

      def parse_signature(handle, index)
        return { index: index, error: 'Null signature handle' } if handle.nil? || handle.null?

        begin
          {
            index: index,
            signer: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_oxide_signature_get_signer(handle)
            ) || 'Unknown',
            timestamp: FFI::Bindings.pdf_oxide_signature_get_timestamp(handle),
            status: STATUS_NAMES[FFI::Bindings.pdf_oxide_signature_get_status(handle)] || :unknown,
            reason: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_oxide_signature_get_reason(handle)
            ) || '',
            location: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_oxide_signature_get_location(handle)
            ) || ''
          }
        ensure
          FFI::Bindings.pdf_oxide_signature_free(handle) unless handle.nil? || handle.null?
        end
      end

      def parse_verification_result(handle)
        return { valid: false, error: 'Null verification handle' } if handle.nil? || handle.null?

        begin
          {
            valid: FFI::Bindings.pdf_oxide_verification_is_valid(handle),
            trusted: FFI::Bindings.pdf_oxide_verification_is_trusted(handle),
            self_signed: FFI::Bindings.pdf_oxide_verification_is_self_signed(handle),
            error_message: FFI::StringMarshaller.from_c_string(
              FFI::Bindings.pdf_oxide_verification_get_error(handle)
            ) || ''
          }
        ensure
          FFI::Bindings.pdf_oxide_verification_free(handle) unless handle.nil? || handle.null?
        end
      end
    end
  end
end
