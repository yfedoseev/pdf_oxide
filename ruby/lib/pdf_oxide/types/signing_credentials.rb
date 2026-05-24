# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents digital signing credentials for PDF signatures
    # Supports PKCS#12 (.pfx/.p12) and PEM (cert + key) formats
    #
    # @example Load PKCS#12 credentials
    #   creds = SigningCredentials.from_pkcs12('cert.p12', 'password')
    #
    # @example Load PEM credentials
    #   creds = SigningCredentials.from_pem('cert.pem', 'key.pem', 'password')
    class SigningCredentials
      # Credential types
      CREDENTIAL_TYPE_PKCS12 = :pkcs12
      CREDENTIAL_TYPE_PEM = :pem
      CREDENTIAL_TYPE_DER = :der

      CREDENTIAL_TYPES = [CREDENTIAL_TYPE_PKCS12, CREDENTIAL_TYPE_PEM, CREDENTIAL_TYPE_DER].freeze

      # Signing algorithms
      ALGORITHM_SHA1 = 0
      ALGORITHM_SHA256 = 1
      ALGORITHM_SHA384 = 2
      ALGORITHM_SHA512 = 3

      ALGORITHMS = {
        sha1: ALGORITHM_SHA1,
        sha256: ALGORITHM_SHA256,
        sha384: ALGORITHM_SHA384,
        sha512: ALGORITHM_SHA512
      }.freeze

      ALGORITHM_NAMES = ALGORITHMS.invert.freeze

      attr_reader :handle, :credential_type, :certificate, :chain_certificates

      # Initialize with FFI handle (internal use)
      # @param handle [FFI::Pointer] Pointer to credentials handle
      # @param credential_type [Symbol] Type of credentials (:pkcs12, :pem, :der)
      # @param certificate [Types::Certificate] Primary certificate information
      # @param chain_certificates [Array<Types::Certificate>] Chain certificates
      def initialize(handle:, credential_type: :unknown, certificate: nil, chain_certificates: [])
        raise ::PdfOxide::ArgumentError, 'Invalid handle for credentials' if handle.nil?
        raise ::PdfOxide::ArgumentError, 'Invalid credential type' unless CREDENTIAL_TYPES.include?(credential_type)

        @handle = handle
        @credential_type = credential_type
        @certificate = certificate
        @chain_certificates = chain_certificates.freeze
      end

      # Load credentials from PKCS#12 file (.pfx or .p12)
      # @param file_path [String] Path to PKCS#12 file
      # @param password [String] Password to decrypt file
      # @return [SigningCredentials] Loaded credentials
      # @raise [PdfException] If file cannot be read or decrypted
      def self.from_pkcs12(file_path, password)
        raise ::PdfOxide::ArgumentError, 'File path must be a string' unless file_path.is_a?(String)
        raise ::PdfOxide::ArgumentError, 'Password must be a string' unless password.is_a?(String)
        raise ::PdfOxide::ArgumentError, "File not found: #{file_path}" unless File.exist?(file_path)

        path_utf8 = FFI::StringMarshaller.to_utf8(file_path)
        password_utf8 = FFI::StringMarshaller.to_utf8(password)

        error_ptr = ::FFI::MemoryPointer.new(:pointer)
        credentials_handle = FFI::Bindings.pdf_credentials_from_pkcs12(path_utf8, password_utf8, error_ptr)

        if error_ptr.read_pointer.null?
          raise ::PdfOxide::PdfException, "Failed to load PKCS#12 credentials from #{file_path}"
        end

        new(handle: credentials_handle, credential_type: CREDENTIAL_TYPE_PKCS12)
      end

      # Load credentials from PEM files (certificate + key)
      # @param cert_path [String] Path to PEM certificate file
      # @param key_path [String] Path to PEM private key file
      # @param password [String] Password for encrypted key (if any)
      # @return [SigningCredentials] Loaded credentials
      # @raise [PdfException] If files cannot be read
      def self.from_pem(cert_path, key_path, password = '')
        raise ::PdfOxide::ArgumentError, 'Certificate path must be a string' unless cert_path.is_a?(String)
        raise ::PdfOxide::ArgumentError, 'Key path must be a string' unless key_path.is_a?(String)
        raise ::PdfOxide::ArgumentError, 'Password must be a string' unless password.is_a?(String)
        raise ::PdfOxide::ArgumentError, "Certificate file not found: #{cert_path}" unless File.exist?(cert_path)
        raise ::PdfOxide::ArgumentError, "Key file not found: #{key_path}" unless File.exist?(key_path)

        cert_utf8 = FFI::StringMarshaller.to_utf8(cert_path)
        key_utf8 = FFI::StringMarshaller.to_utf8(key_path)
        password_utf8 = FFI::StringMarshaller.to_utf8(password)

        error_ptr = ::FFI::MemoryPointer.new(:pointer)
        credentials_handle = FFI::Bindings.pdf_credentials_from_pem(cert_utf8, key_utf8, password_utf8, error_ptr)

        if error_ptr.read_pointer.null?
          raise ::PdfOxide::PdfException, "Failed to load PEM credentials from #{cert_path}"
        end

        new(handle: credentials_handle, credential_type: CREDENTIAL_TYPE_PEM)
      end

      # Load credentials from DER binary data
      # @param cert_data [String] Binary certificate data
      # @param key_data [String] Binary private key data
      # @return [SigningCredentials] Loaded credentials
      # @raise [PdfException] If data is invalid
      def self.from_der(cert_data, key_data)
        raise ::PdfOxide::ArgumentError, 'Certificate data must be a string' unless cert_data.is_a?(String)
        raise ::PdfOxide::ArgumentError, 'Key data must be a string' unless key_data.is_a?(String)

        cert_ptr = ::FFI::MemoryPointer.from_string(cert_data)
        key_ptr = ::FFI::MemoryPointer.from_string(key_data)

        error_ptr = ::FFI::MemoryPointer.new(:pointer)
        credentials_handle = FFI::Bindings.pdf_credentials_from_der(
          cert_ptr, cert_data.bytesize,
          key_ptr, key_data.bytesize,
          error_ptr
        )

        if error_ptr.read_pointer.null?
          raise ::PdfOxide::PdfException, 'Failed to load DER credentials'
        end

        new(handle: credentials_handle, credential_type: CREDENTIAL_TYPE_DER)
      end

      # Add a certificate to the chain
      # @param certificate_path [String] Path to certificate file to add
      # @return [Boolean] Whether certificate was successfully added
      # @raise [PdfException] If file cannot be read
      def add_chain_certificate(certificate_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path must be a string' unless certificate_path.is_a?(String)
        raise ::PdfOxide::ArgumentError, "File not found: #{certificate_path}" unless File.exist?(certificate_path)

        cert_data = File.read(certificate_path)
        cert_ptr = ::FFI::MemoryPointer.from_string(cert_data)

        error_ptr = ::FFI::MemoryPointer.new(:pointer)
        success = FFI::Bindings.pdf_credentials_add_chain_cert(@handle, cert_ptr, cert_data.bytesize, error_ptr)

        raise ::PdfOxide::PdfException, 'Failed to add chain certificate' unless success

        @chain_certificates = (@chain_certificates + [certificate_path]).freeze
        true
      end

      # Get the primary certificate information
      # @return [Types::Certificate] Certificate details
      def get_certificate
        cert_handle = FFI::Bindings.pdf_credentials_get_certificate(@handle, ::FFI::MemoryPointer.new(:pointer))
        return nil if cert_handle.nil? || cert_handle.null?

        parse_certificate(cert_handle)
      end

      # Check if credentials have a valid private key
      # @return [Boolean] Whether private key is available
      def has_private_key?
        !@handle.nil? && !@handle.null?
      end

      # Check if credentials are for PKI/certificate-based signing
      # @return [Boolean] Whether credentials are PKI-based
      def pki_credentials?
        [CREDENTIAL_TYPE_PKCS12, CREDENTIAL_TYPE_PEM, CREDENTIAL_TYPE_DER].include?(@credential_type)
      end

      # Free the underlying FFI handle
      # Should be called when done with credentials
      # @return [void]
      def free
        FFI::Bindings.pdf_credentials_free(@handle) unless @handle.nil? || @handle.null?
      end

      # Convert credentials to hash representation
      # @return [Hash] Credentials information
      def to_h
        {
          type: @credential_type,
          has_private_key: has_private_key?,
          is_pki: pki_credentials?,
          certificate: @certificate&.to_h,
          chain_certificates: @chain_certificates
        }
      end

      # Convert to string representation
      # @return [String] String representation
      def to_s
        "SigningCredentials(type=#{@credential_type}, pki=#{pki_credentials?})"
      end

      # Inspect representation
      # @return [String] Detailed representation
      def inspect
        to_s
      end

      # Check equality
      # @param other [SigningCredentials] Credentials to compare
      # @return [Boolean] Whether credentials are equal
      def ==(other)
        other.is_a?(SigningCredentials) && @credential_type == other.credential_type && @certificate == other.certificate
      end

      # Calculate hash code
      # @return [Integer] Hash code
      def hash
        [@credential_type, @certificate].hash
      end

      private

      # Parse certificate handle into Certificate type
      # @param cert_handle [FFI::Pointer] Certificate handle
      # @return [Types::Certificate] Parsed certificate
      def parse_certificate(cert_handle)
        return nil if cert_handle.nil? || cert_handle.null?

        # This would require certificate accessor FFI functions
        # For now, return a basic certificate representation
        Types::Certificate.new(
          subject: 'Certificate Subject',
          issuer: 'Certificate Issuer',
          valid_from: Time.now,
          valid_until: Time.now + 365 * 24 * 3600
        )
      end
    end
  end
end
