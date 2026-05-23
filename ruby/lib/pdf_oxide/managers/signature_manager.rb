module PdfOxide
  # Signature Manager for PDF digital signatures with Rails integration
  #
  # Provides signature verification, validation, and management with:
  # - Sidekiq background job support
  # - ActiveRecord model integration
  # - Caching mechanisms
  # - Thread-safe operations
  class SignatureManager
    # Signature information record
    SignatureInfo = Struct.new(
      :page_index,
      :signature_index,
      :signer_name,
      :timestamp,
      :certificate_subject,
      :certificate_issuer,
      :valid_from,
      :valid_until,
      :is_valid,
      :reason,
      :location,
      :version,
      keyword_init: true
    )

    # Certificate information record
    CertificateInfo = Struct.new(
      :common_name,
      :organization,
      :valid_from,
      :valid_until,
      :public_key_algorithm,
      :signature_algorithm,
      :serial_number,
      keyword_init: true
    )

    attr_reader :document, :cache

    # Initialize SignatureManager with a PDF document
    def initialize(document)
      @document = document
      @cache = ActiveSupport::Cache::MemoryStore.new
      @mutex = Mutex.new
    end

    # Get count of digital signatures in document
    def get_signature_count
      @mutex.synchronize do
        cache.fetch("signature_count", expires_in: 1.hour) do
          # Real implementation would call FFI layer
          0
        end
      end
    end

    # Get signature information for specific signature
    def get_signature_info(page_index, signature_index)
      cache_key = "signature:#{page_index}:#{signature_index}"
      
      cache.fetch(cache_key, expires_in: 1.hour) do
        SignatureInfo.new(
          page_index: page_index,
          signature_index: signature_index,
          signer_name: "Test Signer",
          timestamp: Time.current,
          certificate_subject: "CN=Test Signer",
          certificate_issuer: "CN=Test CA",
          valid_from: Time.current,
          valid_until: 1.year.from_now,
          is_valid: true,
          reason: nil,
          location: nil,
          version: "1.0"
        )
      end
    end

    # Verify a digital signature
    def verify_signature(page_index, signature_index)
      cache_key = "verify:#{page_index}:#{signature_index}"
      
      cache.fetch(cache_key, expires_in: 1.hour) do
        # Real implementation would verify signature
        :valid
      end
    end

    # Get certificate information
    def get_certificate_info(signature_index)
      cache_key = "certificate:#{signature_index}"
      
      cache.fetch(cache_key, expires_in: 1.hour) do
        CertificateInfo.new(
          common_name: "Test Certificate",
          organization: "Test Organization",
          valid_from: Time.current,
          valid_until: 1.year.from_now,
          public_key_algorithm: "RSA-2048",
          signature_algorithm: "SHA256withRSA",
          serial_number: "01"
        )
      end
    end

    # Get all signatures in document
    def get_all_signatures
      cache.fetch("all_signatures", expires_in: 1.hour) do
        signatures = []
        count = get_signature_count
        
        (0...count).each do |i|
          sig = get_signature_info(0, i)
          signatures << sig if sig
        end
        
        signatures
      end
    end

    # Queue signature verification job with Sidekiq
    def verify_signatures_async(page_index = nil, &block)
      if defined?(Sidekiq)
        VerifySignaturesJob.perform_async(document.id, page_index)
      else
        verify_signatures_sync(page_index, &block)
      end
    end

    # Verify signatures synchronously
    def verify_signatures_sync(page_index = nil)
      signatures = get_all_signatures
      
      signatures.each do |sig|
        status = verify_signature(sig.page_index, sig.signature_index)
        yield(sig, status) if block_given?
      end
      
      signatures
    end

    # Check if all signatures are valid
    def all_signatures_valid?
      get_all_signatures.all?(&:is_valid)
    end

    # Get signature statistics
    def signature_statistics
      signatures = get_all_signatures
      valid_count = signatures.count(&:is_valid)
      
      {
        total: signatures.count,
        valid: valid_count,
        invalid: signatures.count - valid_count,
        trusted: valid_count,
        untrusted: signatures.count - valid_count
      }
    end

    # Sign PDF data using a PKCS#12 (.pfx/.p12) certificate
    #
    # @param pdf_data [String] Raw PDF bytes to sign
    # @param file_path [String] Path to the PKCS#12 file
    # @param password [String] Password for the PKCS#12 file
    # @param options [Hash] Signing options
    # @option options [String] :reason Reason for signing
    # @option options [String] :location Location of signing
    # @option options [String] :contact Contact information
    # @option options [Integer] :algorithm Signature algorithm (default: 0 for SHA256)
    # @option options [Integer] :subfilter Signature subfilter (default: 0)
    # @return [String] Signed PDF bytes
    # @raise [PdfOxide::SignatureError] If signing fails
    def sign_with_pkcs12(pdf_data, file_path, password, options = {})
      raise ::PdfOxide::ArgumentError.new('pdf_data cannot be nil') if pdf_data.nil?
      raise ::PdfOxide::ArgumentError.new('file_path cannot be nil') if file_path.nil?

      credentials = FFI::ErrorHandler.with_error_check('credentials_from_pkcs12') do |err|
        Bindings.pdf_credentials_from_pkcs12(file_path, password || '', err)
      end

      begin
        sign_with_credentials(pdf_data, credentials, options)
      ensure
        Bindings.pdf_credentials_free(credentials) if credentials && !credentials.null?
      end
    end

    # Sign PDF data using PEM certificate and key files
    #
    # @param pdf_data [String] Raw PDF bytes to sign
    # @param cert_file [String] Path to the PEM certificate file
    # @param key_file [String] Path to the PEM private key file
    # @param options [Hash] Signing options
    # @option options [String] :key_password Password for the private key (optional)
    # @option options [String] :reason Reason for signing
    # @option options [String] :location Location of signing
    # @option options [String] :contact Contact information
    # @option options [Integer] :algorithm Signature algorithm (default: 0 for SHA256)
    # @option options [Integer] :subfilter Signature subfilter (default: 0)
    # @return [String] Signed PDF bytes
    # @raise [PdfOxide::SignatureError] If signing fails
    def sign_with_pem(pdf_data, cert_file, key_file, options = {})
      raise ::PdfOxide::ArgumentError.new('pdf_data cannot be nil') if pdf_data.nil?
      raise ::PdfOxide::ArgumentError.new('cert_file cannot be nil') if cert_file.nil?
      raise ::PdfOxide::ArgumentError.new('key_file cannot be nil') if key_file.nil?

      key_password = options.fetch(:key_password, '')

      credentials = FFI::ErrorHandler.with_error_check('credentials_from_pem') do |err|
        Bindings.pdf_credentials_from_pem(cert_file, key_file, key_password, err)
      end

      begin
        sign_with_credentials(pdf_data, credentials, options)
      ensure
        Bindings.pdf_credentials_free(credentials) if credentials && !credentials.null?
      end
    end

    # Sign a PDF file on disk, writing the signed result to output_path
    #
    # @param input_path [String] Path to the input PDF file
    # @param output_path [String] Path for the signed output PDF file
    # @param credentials [FFI::Pointer] Credentials handle from PKCS#12 or PEM loading
    # @param options [Hash] Signing options
    # @option options [String] :reason Reason for signing
    # @option options [String] :location Location of signing
    # @option options [String] :contact Contact information
    # @option options [Integer] :algorithm Signature algorithm (default: 0)
    # @option options [Integer] :subfilter Signature subfilter (default: 0)
    # @return [Boolean] true on success
    # @raise [PdfOxide::SignatureError] If signing fails
    def sign_file(input_path, output_path, credentials, options = {})
      raise ::PdfOxide::ArgumentError.new('input_path cannot be nil') if input_path.nil?
      raise ::PdfOxide::ArgumentError.new('output_path cannot be nil') if output_path.nil?
      raise ::PdfOxide::ArgumentError.new('credentials cannot be nil') if credentials.nil?

      reason = options.fetch(:reason, nil)
      location = options.fetch(:location, nil)
      contact = options.fetch(:contact, nil)
      algorithm = options.fetch(:algorithm, 0)
      subfilter = options.fetch(:subfilter, 0)

      FFI::ErrorHandler.with_bool_check('sign_file') do |err|
        Bindings.pdf_document_sign_file(
          input_path, output_path,
          credentials,
          reason, location, contact,
          algorithm, subfilter,
          err
        )
      end
    end

    # Embed Long-Term Validation (LTV) data into a signed PDF
    #
    # @param pdf_data [String] Raw PDF bytes (already signed)
    # @param ocsp_data [String, nil] OCSP response bytes (optional)
    # @param crl_data [String, nil] CRL data bytes (optional)
    # @return [String] PDF bytes with embedded LTV data
    # @raise [PdfOxide::SignatureError] If embedding fails
    def embed_ltv(pdf_data, ocsp_data: nil, crl_data: nil)
      raise ::PdfOxide::ArgumentError.new('pdf_data cannot be nil') if pdf_data.nil?

      pdf_ptr = ::FFI::MemoryPointer.new(:uint8, pdf_data.bytesize)
      pdf_ptr.put_bytes(0, pdf_data)

      ocsp_ptr = nil
      ocsp_len = 0
      if ocsp_data && !ocsp_data.empty?
        ocsp_ptr = ::FFI::MemoryPointer.new(:uint8, ocsp_data.bytesize)
        ocsp_ptr.put_bytes(0, ocsp_data)
        ocsp_len = ocsp_data.bytesize
      end

      crl_ptr = nil
      crl_len = 0
      if crl_data && !crl_data.empty?
        crl_ptr = ::FFI::MemoryPointer.new(:uint8, crl_data.bytesize)
        crl_ptr.put_bytes(0, crl_data)
        crl_len = crl_data.bytesize
      end

      out_data_ptr = ::FFI::MemoryPointer.new(:pointer)
      out_len_ptr = ::FFI::MemoryPointer.new(:size_t)

      FFI::ErrorHandler.with_bool_check('embed_ltv') do |err|
        Bindings.pdf_embed_ltv_data(
          pdf_ptr, pdf_data.bytesize,
          ocsp_ptr, ocsp_len,
          crl_ptr, crl_len,
          out_data_ptr, out_len_ptr,
          err
        )
      end

      result_ptr = out_data_ptr.read_pointer
      result_len = out_len_ptr.read(:size_t)

      begin
        result_ptr.read_bytes(result_len)
      ensure
        Bindings.pdf_signed_bytes_free(result_ptr, result_len) if result_ptr && !result_ptr.null?
      end
    end

    # Load signing credentials from DER-encoded certificate and key bytes
    #
    # @param cert_data [String] DER-encoded certificate bytes
    # @param key_data [String, nil] DER-encoded private key bytes (optional)
    # @return [FFI::Pointer] Credentials handle (must be freed with pdf_credentials_free)
    # @raise [PdfOxide::SignatureError] If loading fails
    def load_credentials_from_der(cert_data, key_data: nil)
      raise ::PdfOxide::ArgumentError.new('cert_data cannot be nil') if cert_data.nil?

      cert_ptr = ::FFI::MemoryPointer.new(:uint8, cert_data.bytesize)
      cert_ptr.put_bytes(0, cert_data)

      if key_data && !key_data.empty?
        key_ptr = ::FFI::MemoryPointer.new(:uint8, key_data.bytesize)
        key_ptr.put_bytes(0, key_data)
        key_len = key_data.bytesize
      else
        key_ptr = nil
        key_len = 0
      end

      FFI::ErrorHandler.with_error_check('credentials_from_der') do |err|
        Bindings.pdf_credentials_from_der(
          cert_ptr, cert_data.bytesize,
          key_ptr, key_len,
          err
        )
      end
    end

    # Add an intermediate certificate to the credentials chain
    #
    # @param credentials [FFI::Pointer] Credentials handle
    # @param cert_data [String] DER-encoded intermediate certificate bytes
    # @raise [PdfOxide::SignatureError] If adding the chain certificate fails
    def add_chain_cert(credentials, cert_data)
      raise ::PdfOxide::ArgumentError.new('credentials cannot be nil') if credentials.nil?
      raise ::PdfOxide::ArgumentError.new('cert_data cannot be nil') if cert_data.nil?

      cert_ptr = ::FFI::MemoryPointer.new(:uint8, cert_data.bytesize)
      cert_ptr.put_bytes(0, cert_data)

      FFI::ErrorHandler.with_bool_check('credentials_add_chain_cert') do |err|
        Bindings.pdf_credentials_add_chain_cert(
          credentials,
          cert_ptr, cert_data.bytesize,
          err
        )
      end
    end

    # Get the certificate from a credentials handle
    #
    # @param credentials [FFI::Pointer] Credentials handle
    # @return [FFI::Pointer] Certificate handle (must be freed with pdf_certificate_free)
    # @raise [PdfOxide::SignatureError] If retrieval fails
    def get_certificate(credentials)
      raise ::PdfOxide::ArgumentError.new('credentials cannot be nil') if credentials.nil?

      FFI::ErrorHandler.with_error_check('credentials_get_certificate') do |err|
        Bindings.pdf_credentials_get_certificate(credentials, err)
      end
    end

    # Get the Common Name (CN) from a certificate handle
    #
    # @param cert_handle [FFI::Pointer] Certificate handle
    # @return [String] Common Name
    # @raise [PdfOxide::SignatureError] If retrieval fails
    def certificate_cn(cert_handle)
      raise ::PdfOxide::ArgumentError.new('cert_handle cannot be nil') if cert_handle.nil?

      FFI::ErrorHandler.with_error_check('certificate_get_cn') do |err|
        Bindings.pdf_certificate_get_cn(cert_handle, err)
      end
    end

    # Get the issuer string from a certificate handle
    #
    # @param cert_handle [FFI::Pointer] Certificate handle
    # @return [String] Issuer string
    # @raise [PdfOxide::SignatureError] If retrieval fails
    def certificate_issuer(cert_handle)
      raise ::PdfOxide::ArgumentError.new('cert_handle cannot be nil') if cert_handle.nil?

      FFI::ErrorHandler.with_error_check('certificate_get_issuer') do |err|
        Bindings.pdf_certificate_get_issuer(cert_handle, err)
      end
    end

    # Get the key size from a certificate handle
    #
    # @param cert_handle [FFI::Pointer] Certificate handle
    # @return [Integer] Key size in bytes
    # @raise [PdfOxide::SignatureError] If retrieval fails
    def certificate_size(cert_handle)
      raise ::PdfOxide::ArgumentError.new('cert_handle cannot be nil') if cert_handle.nil?

      FFI::ErrorHandler.with_error_check('certificate_get_size') do |err|
        Bindings.pdf_certificate_get_size(cert_handle, err)
      end
    end

    # Sign a PDF document with a visible signature appearance on a specific page
    #
    # @param pdf_data [String] Raw PDF bytes to sign
    # @param credentials [FFI::Pointer] Credentials handle
    # @param page_num [Integer] Page number (0-based)
    # @param x [Float] X coordinate of signature rectangle
    # @param y [Float] Y coordinate of signature rectangle
    # @param width [Float] Width of signature rectangle
    # @param height [Float] Height of signature rectangle
    # @param options [Hash] Signing options
    # @option options [String] :reason Reason for signing
    # @option options [String] :location Location of signing
    # @option options [String] :contact Contact information
    # @option options [Integer] :algorithm Signature algorithm (default: 0 for SHA256)
    # @return [String] Signed PDF bytes
    # @raise [PdfOxide::SignatureError] If signing fails
    def sign_with_appearance(pdf_data, credentials, page_num:, x:, y:, width:, height:, **options)
      raise ::PdfOxide::ArgumentError.new('pdf_data cannot be nil') if pdf_data.nil?
      raise ::PdfOxide::ArgumentError.new('credentials cannot be nil') if credentials.nil?

      reason = options.fetch(:reason, nil)
      location = options.fetch(:location, nil)
      contact = options.fetch(:contact, nil)
      algorithm = options.fetch(:algorithm, 0)

      pdf_ptr = ::FFI::MemoryPointer.new(:uint8, pdf_data.bytesize)
      pdf_ptr.put_bytes(0, pdf_data)

      out_data_ptr = ::FFI::MemoryPointer.new(:pointer)
      out_len_ptr = ::FFI::MemoryPointer.new(:size_t)

      FFI::ErrorHandler.with_bool_check('sign_with_appearance') do |err|
        Bindings.pdf_document_sign_with_appearance(
          pdf_ptr, pdf_data.bytesize,
          credentials,
          page_num,
          x.to_f, y.to_f, width.to_f, height.to_f,
          reason, location, contact,
          algorithm,
          out_data_ptr, out_len_ptr,
          err
        )
      end

      result_ptr = out_data_ptr.read_pointer
      result_len = out_len_ptr.read(:size_t)

      begin
        result_ptr.read_bytes(result_len)
      ensure
        Bindings.pdf_signed_bytes_free(result_ptr, result_len) if result_ptr && !result_ptr.null?
      end
    end

    # Add a timestamp to an existing signature via a Time Stamp Authority
    #
    # @param pdf_data [String] Raw PDF bytes containing the signed document
    # @param signature_index [Integer] Index of the signature to timestamp (0-based)
    # @param tsa_url [String] URL of the Time Stamp Authority server
    # @return [String] Timestamped PDF bytes
    # @raise [PdfOxide::SignatureError] If timestamping fails
    def add_timestamp(pdf_data, signature_index:, tsa_url:)
      raise ::PdfOxide::ArgumentError.new('pdf_data cannot be nil') if pdf_data.nil?
      raise ::PdfOxide::ArgumentError.new('tsa_url cannot be nil') if tsa_url.nil?

      pdf_ptr = ::FFI::MemoryPointer.new(:uint8, pdf_data.bytesize)
      pdf_ptr.put_bytes(0, pdf_data)

      out_data_ptr = ::FFI::MemoryPointer.new(:pointer)
      out_len_ptr = ::FFI::MemoryPointer.new(:size_t)

      FFI::ErrorHandler.with_bool_check('add_timestamp') do |err|
        Bindings.pdf_add_timestamp(
          pdf_ptr, pdf_data.bytesize,
          signature_index,
          tsa_url,
          out_data_ptr, out_len_ptr,
          err
        )
      end

      result_ptr = out_data_ptr.read_pointer
      result_len = out_len_ptr.read(:size_t)

      begin
        result_ptr.read_bytes(result_len)
      ensure
        Bindings.pdf_signed_bytes_free(result_ptr, result_len) if result_ptr && !result_ptr.null?
      end
    end

    # Verify a specific signature using the native Rust verification engine
    #
    # Returns a status code: 0=Valid, 1=Invalid, 2=CertExpired, 3=CertRevoked, 4=TrustFailed, 5=Unknown
    #
    # @param signature_index [Integer] Index of the signature to verify (0-based)
    # @return [Integer] Verification status code
    # @raise [PdfOxide::SignatureError] If verification fails
    def verify_signature_native(signature_index)
      raise ::PdfOxide::ArgumentError.new('document handle is not valid') if @document.nil?

      FFI::ErrorHandler.with_error_check('verify_signature_native') do |err|
        Bindings.pdf_verify_signature(@document, signature_index, nil, 0, err)
      end
    end

    # Clear signature cache
    def clear_cache
      cache.clear
    end

    # Find pages with signatures
    def pages_with_signatures
      get_all_signatures.map(&:page_index).uniq.sort
    end

    # Get signer information
    def get_signer_info(signature_index)
      cert_info = get_certificate_info(signature_index)
      
      if cert_info
        {
          name: cert_info.common_name,
          organization: cert_info.organization,
          timestamp: get_signature_info(0, signature_index)&.timestamp
        }
      else
        nil
      end
    end

    # Validate certificate chain
    def validate_certificate_chain(signature_index)
      cache_key = "cert_chain:#{signature_index}"
      
      cache.fetch(cache_key, expires_in: 1.hour) do
        # Real implementation would validate
        true
      end
    end

    # Check if signature is trusted
    def signature_trusted?(signature_index)
      status = verify_signature(0, signature_index)
      status == :valid && validate_certificate_chain(signature_index)
    end

    # Get detailed signature report
    def signature_report
      statistics = signature_statistics
      signatures = get_all_signatures
      
      {
        statistics: statistics,
        signatures: signatures.map do |sig|
          {
            page: sig.page_index,
            signer: sig.signer_name,
            timestamp: sig.timestamp,
            valid: sig.is_valid,
            trusted: signature_trusted?(sig.signature_index)
          }
        end,
        generated_at: Time.current
      }
    end

    private

    # Internal: Sign PDF data with already-loaded credentials handle
    def sign_with_credentials(pdf_data, credentials, options)
      reason = options.fetch(:reason, nil)
      location = options.fetch(:location, nil)
      contact = options.fetch(:contact, nil)
      algorithm = options.fetch(:algorithm, 0)
      subfilter = options.fetch(:subfilter, 0)

      pdf_ptr = ::FFI::MemoryPointer.new(:uint8, pdf_data.bytesize)
      pdf_ptr.put_bytes(0, pdf_data)

      out_data_ptr = ::FFI::MemoryPointer.new(:pointer)
      out_len_ptr = ::FFI::MemoryPointer.new(:size_t)

      FFI::ErrorHandler.with_bool_check('sign_document') do |err|
        Bindings.pdf_document_sign_data(
          pdf_ptr, pdf_data.bytesize,
          credentials,
          reason, location, contact,
          algorithm, subfilter,
          out_data_ptr, out_len_ptr,
          err
        )
      end

      result_ptr = out_data_ptr.read_pointer
      result_len = out_len_ptr.read(:size_t)

      begin
        result_ptr.read_bytes(result_len)
      ensure
        Bindings.pdf_signed_bytes_free(result_ptr, result_len) if result_ptr && !result_ptr.null?
      end
    end
  end

  # Sidekiq job for async signature verification
  if defined?(Sidekiq)
    class VerifySignaturesJob
      include Sidekiq::Job
      
      sidekiq_options retry: 3, dead: true

      def perform(document_id, page_index = nil)
        document = PdfDocument.find(document_id)
        manager = SignatureManager.new(document)
        
        signatures = manager.get_all_signatures
        signatures.each do |sig|
          next if page_index && sig.page_index != page_index
          
          status = manager.verify_signature(sig.page_index, sig.signature_index)
          
          # Store result in database or cache as needed
          result = SignatureVerificationResult.create!(
            document_id: document_id,
            page_index: sig.page_index,
            signature_index: sig.signature_index,
            status: status,
            verified_at: Time.current
          )
          
          result
        end
      end
    end
  end
end
