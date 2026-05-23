# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for certificate operations
    # Handles loading and managing digital certificates for signatures
    class Certificate < Base
      # Load certificate from file
      # @param cert_path [String] Path to certificate file
      # @return [Hash] Certificate information
      def load_certificate(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?
        raise ::PdfOxide::ArgumentError, 'Certificate file not found' unless File.exist?(cert_path)

        cert_path_utf8 = FFI::StringMarshaller.to_utf8(cert_path)

        # Read file bytes
        cert_bytes = File.read(cert_path)

        load_certificate_from_bytes(cert_bytes)
      end

      # Load certificate from bytes
      # @param cert_data [String, Bytes] Certificate data
      # @return [Hash] Certificate information
      def load_certificate_from_bytes(cert_data)
        raise ::PdfOxide::ArgumentError, 'Certificate data cannot be empty' if cert_data.nil? || cert_data.empty?

        cert_ptr = with_error_check('load_certificate_from_bytes') do |error_ptr|
          FFI::Bindings.pdf_certificate_load_from_bytes(cert_data, cert_data.bytesize, error_ptr)
        end

        parse_certificate(cert_ptr)
      end

      # Get certificate information
      # @param cert_path [String] Path to certificate file
      # @return [Hash] Complete certificate details
      def get_certificate_info(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert
      end

      # Validate certificate
      # @param cert_path [String] Path to certificate file
      # @return [Boolean] Whether certificate is valid
      def is_certificate_valid?(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:valid] && !cert[:expired]
      end

      # Check if certificate is expired
      # @param cert_path [String] Path to certificate file
      # @return [Boolean] Whether certificate is expired
      def is_certificate_expired?(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:expired]
      end

      # Get certificate subject
      # @param cert_path [String] Path to certificate file
      # @return [String] Certificate subject
      def get_certificate_subject(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:subject] || ''
      end

      # Get certificate issuer
      # @param cert_path [String] Path to certificate file
      # @return [String] Certificate issuer
      def get_certificate_issuer(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:issuer] || ''
      end

      # Get certificate validity period
      # @param cert_path [String] Path to certificate file
      # @return [Hash] Valid from and to dates
      def get_certificate_validity(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)

        valid_from_time = begin
          Time.at(cert[:valid_from_timestamp])
        rescue
          nil
        end

        valid_to_time = begin
          Time.at(cert[:valid_to_timestamp])
        rescue
          nil
        end

        {
          valid_from: cert[:valid_from],
          valid_to: cert[:valid_to],
          valid_from_time: valid_from_time,
          valid_to_time: valid_to_time
        }
      end

      # Get certificate serial number
      # @param cert_path [String] Path to certificate file
      # @return [String] Serial number
      def get_certificate_serial_number(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:serial_number] || ''
      end

      # Get certificate thumbprint
      # @param cert_path [String] Path to certificate file
      # @return [String] Certificate thumbprint
      def get_certificate_thumbprint(cert_path)
        raise ::PdfOxide::ArgumentError, 'Certificate path cannot be empty' if cert_path.nil? || cert_path.empty?

        cert = load_certificate(cert_path)
        cert[:thumbprint] || ''
      end

      # List all available certificates
      # @return [Array<Hash>] List of certificates
      def list_certificates
        # This would require system certificate store access
        []
      end

      # Certificate statistics
      # @return [Hash] Certificate statistics
      def certificate_statistics
        {
          total_certificates: 0,
          valid_certificates: 0,
          expired_certificates: 0,
          timestamp: Time.now.to_i
        }
      end

      private

      def parse_certificate(cert_ptr)
        return {} if cert_ptr.nil? || cert_ptr.null?

        begin
          subject_ptr = FFI::Bindings.pdf_certificate_get_subject(cert_ptr)
          issuer_ptr = FFI::Bindings.pdf_certificate_get_issuer(cert_ptr)
          serial_ptr = FFI::Bindings.pdf_certificate_get_serial_number(cert_ptr)
          thumbprint_ptr = FFI::Bindings.pdf_certificate_get_thumbprint(cert_ptr)

          subject = FFI::StringMarshaller.read_c_string(subject_ptr) || ''
          issuer = FFI::StringMarshaller.read_c_string(issuer_ptr) || ''
          serial = FFI::StringMarshaller.read_c_string(serial_ptr) || ''
          thumbprint = FFI::StringMarshaller.read_c_string(thumbprint_ptr) || ''

          valid_from = FFI::Bindings.pdf_certificate_get_valid_from(cert_ptr)
          valid_to = FFI::Bindings.pdf_certificate_get_valid_to(cert_ptr)

          now = Time.now.to_i
          expired = now > valid_to

          {
            subject: subject,
            issuer: issuer,
            serial_number: serial,
            thumbprint: thumbprint,
            valid_from: valid_from,
            valid_to: valid_to,
            valid_from_timestamp: valid_from,
            valid_to_timestamp: valid_to,
            valid: true,
            expired: expired
          }
        ensure
          FFI::Bindings.pdf_certificate_free(cert_ptr) unless cert_ptr.nil? || cert_ptr.null?
        end
      end
    end
  end
end
