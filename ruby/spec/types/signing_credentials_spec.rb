# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Types::SigningCredentials do
  let(:mock_handle) { double(null?: false) }
  let(:temp_dir) { Dir.tmpdir }
  let(:pkcs12_file) { File.join(temp_dir, 'test.p12') }
  let(:cert_file) { File.join(temp_dir, 'test.crt') }
  let(:key_file) { File.join(temp_dir, 'test.key') }

  before do
    # Create dummy files for testing
    File.write(pkcs12_file, 'fake pkcs12 data') unless File.exist?(pkcs12_file)
    File.write(cert_file, 'fake cert data') unless File.exist?(cert_file)
    File.write(key_file, 'fake key data') unless File.exist?(key_file)
  end

  after do
    File.delete(pkcs12_file) if File.exist?(pkcs12_file)
    File.delete(cert_file) if File.exist?(cert_file)
    File.delete(key_file) if File.exist?(key_file)
  end

  describe 'credential type constants' do
    it 'defines credential type constants' do
      expect(described_class::CREDENTIAL_TYPE_PKCS12).to eq(:pkcs12)
      expect(described_class::CREDENTIAL_TYPE_PEM).to eq(:pem)
      expect(described_class::CREDENTIAL_TYPE_DER).to eq(:der)
    end

    it 'defines valid credential types' do
      expect(described_class::CREDENTIAL_TYPES).to include(:pkcs12, :pem, :der)
    end
  end

  describe 'algorithm constants' do
    it 'defines algorithm constants' do
      expect(described_class::ALGORITHM_SHA1).to eq(0)
      expect(described_class::ALGORITHM_SHA256).to eq(1)
      expect(described_class::ALGORITHM_SHA384).to eq(2)
      expect(described_class::ALGORITHM_SHA512).to eq(3)
    end

    it 'defines algorithm map' do
      expect(described_class::ALGORITHMS[:sha256]).to eq(1)
      expect(described_class::ALGORITHMS[:sha512]).to eq(3)
    end
  end

  describe '#initialize' do
    it 'creates credentials with valid handle' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      expect(creds.handle).to eq(mock_handle)
      expect(creds.credential_type).to eq(:pkcs12)
    end

    it 'validates handle is not nil' do
      expect { described_class.new(handle: nil, credential_type: :pkcs12) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates credential type' do
      expect { described_class.new(handle: mock_handle, credential_type: :invalid) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'initializes with empty chain certificates by default' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      expect(creds.chain_certificates).to eq([])
      expect(creds.chain_certificates).to be_frozen
    end

    it 'accepts optional certificate and chain' do
      cert = double
      chain = [double]
      creds = described_class.new(
        handle: mock_handle,
        credential_type: :pkcs12,
        certificate: cert,
        chain_certificates: chain
      )
      expect(creds.certificate).to eq(cert)
      expect(creds.chain_certificates).to eq(chain)
    end
  end

  describe '.from_pkcs12' do
    it 'loads credentials from PKCS#12 file' do
      allow(FFI::Bindings).to receive(:pdf_credentials_from_pkcs12).and_return(mock_handle)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('path')

      creds = described_class.from_pkcs12(pkcs12_file, 'password')
      expect(creds.credential_type).to eq(:pkcs12)
      expect(creds.handle).to eq(mock_handle)
    end

    it 'validates file path is a string' do
      expect { described_class.from_pkcs12(123, 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates password is a string' do
      expect { described_class.from_pkcs12(pkcs12_file, 123) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates file exists' do
      expect { described_class.from_pkcs12('/nonexistent/file.p12', 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error if FFI call fails' do
      allow(FFI::Bindings).to receive(:pdf_credentials_from_pkcs12).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('path')

      expect { described_class.from_pkcs12(pkcs12_file, 'password') }
        .to raise_error(PdfOxide::PdfException)
    end
  end

  describe '.from_pem' do
    it 'loads credentials from PEM files' do
      allow(FFI::Bindings).to receive(:pdf_credentials_from_pem).and_return(mock_handle)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('path')

      creds = described_class.from_pem(cert_file, key_file, 'password')
      expect(creds.credential_type).to eq(:pem)
      expect(creds.handle).to eq(mock_handle)
    end

    it 'accepts empty password for unencrypted keys' do
      allow(FFI::Bindings).to receive(:pdf_credentials_from_pem).and_return(mock_handle)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('path')

      creds = described_class.from_pem(cert_file, key_file)
      expect(creds.credential_type).to eq(:pem)
    end

    it 'validates certificate path is a string' do
      expect { described_class.from_pem(123, key_file, 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates key path is a string' do
      expect { described_class.from_pem(cert_file, 123, 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates certificate file exists' do
      expect { described_class.from_pem('/nonexistent/cert.pem', key_file, 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates key file exists' do
      expect { described_class.from_pem(cert_file, '/nonexistent/key.pem', 'password') }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '.from_der' do
    it 'loads credentials from DER binary data' do
      cert_data = 'binary cert data'
      key_data = 'binary key data'

      allow(FFI::Bindings).to receive(:pdf_credentials_from_der).and_return(mock_handle)

      creds = described_class.from_der(cert_data, key_data)
      expect(creds.credential_type).to eq(:der)
      expect(creds.handle).to eq(mock_handle)
    end

    it 'validates certificate data is a string' do
      expect { described_class.from_der(123, 'key') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates key data is a string' do
      expect { described_class.from_der('cert', 123) }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#add_chain_certificate' do
    let(:creds) { described_class.new(handle: mock_handle, credential_type: :pkcs12) }
    let(:chain_cert_file) { File.join(temp_dir, 'chain.crt') }

    before do
      File.write(chain_cert_file, 'chain cert data') unless File.exist?(chain_cert_file)
    end

    after do
      File.delete(chain_cert_file) if File.exist?(chain_cert_file)
    end

    it 'adds a certificate to the chain' do
      allow(FFI::Bindings).to receive(:pdf_credentials_add_chain_cert).and_return(true)

      result = creds.add_chain_certificate(chain_cert_file)
      expect(result).to be true
      expect(creds.chain_certificates).to include(chain_cert_file)
    end

    it 'validates path is a string' do
      expect { creds.add_chain_certificate(123) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates file exists' do
      expect { creds.add_chain_certificate('/nonexistent/chain.crt') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error if FFI call fails' do
      allow(FFI::Bindings).to receive(:pdf_credentials_add_chain_cert).and_return(false)

      expect { creds.add_chain_certificate(chain_cert_file) }
        .to raise_error(PdfOxide::PdfException)
    end
  end

  describe '#get_certificate' do
    let(:creds) { described_class.new(handle: mock_handle, credential_type: :pkcs12) }

    it 'returns certificate information' do
      allow(FFI::Bindings).to receive(:pdf_credentials_get_certificate).and_return(mock_handle)

      cert = creds.get_certificate
      expect(cert).to be_a(PdfOxide::Types::Certificate)
    end

    it 'returns nil for null certificate handle' do
      null_handle = double(null?: true)
      allow(FFI::Bindings).to receive(:pdf_credentials_get_certificate).and_return(null_handle)

      cert = creds.get_certificate
      expect(cert).to be_nil
    end
  end

  describe '#has_private_key?' do
    it 'returns true when credentials have valid handle' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      expect(creds.has_private_key?).to be true
    end

    it 'returns false for null handle' do
      null_handle = double(null?: true)
      creds = described_class.new(handle: null_handle, credential_type: :pkcs12)
      expect(creds.has_private_key?).to be false
    end
  end

  describe '#pki_credentials?' do
    it 'returns true for PKCS#12 credentials' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      expect(creds.pki_credentials?).to be true
    end

    it 'returns true for PEM credentials' do
      creds = described_class.new(handle: mock_handle, credential_type: :pem)
      expect(creds.pki_credentials?).to be true
    end

    it 'returns true for DER credentials' do
      creds = described_class.new(handle: mock_handle, credential_type: :der)
      expect(creds.pki_credentials?).to be true
    end
  end

  describe '#free' do
    it 'frees the FFI handle' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      allow(FFI::Bindings).to receive(:pdf_credentials_free)

      creds.free
      expect(FFI::Bindings).to have_received(:pdf_credentials_free)
    end

    it 'handles null handle gracefully' do
      null_handle = double(null?: true)
      creds = described_class.new(handle: null_handle, credential_type: :pkcs12)
      allow(FFI::Bindings).to receive(:pdf_credentials_free)

      creds.free
      expect(FFI::Bindings).not_to have_received(:pdf_credentials_free)
    end
  end

  describe '#to_h' do
    it 'converts credentials to hash' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)

      hash = creds.to_h
      expect(hash[:type]).to eq(:pkcs12)
      expect(hash[:has_private_key]).to be true
      expect(hash[:is_pki]).to be true
    end
  end

  describe '#to_s' do
    it 'returns string representation' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      str = creds.to_s

      expect(str).to include('SigningCredentials')
      expect(str).to include('pkcs12')
    end
  end

  describe '#==' do
    it 'compares credentials for equality' do
      creds1 = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      creds2 = described_class.new(handle: mock_handle, credential_type: :pkcs12)

      expect(creds1).to eq(creds2)
    end

    it 'returns false for different credential types' do
      creds1 = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      creds2 = described_class.new(handle: mock_handle, credential_type: :pem)

      expect(creds1).not_to eq(creds2)
    end

    it 'returns false when comparing to other types' do
      creds = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      expect(creds).not_to eq('string')
      expect(creds).not_to eq(123)
    end
  end

  describe '#hash' do
    it 'generates consistent hash codes' do
      creds1 = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      creds2 = described_class.new(handle: mock_handle, credential_type: :pkcs12)

      expect(creds1.hash).to eq(creds2.hash)
    end

    it 'allows use in hash collections' do
      creds1 = described_class.new(handle: mock_handle, credential_type: :pkcs12)
      creds2 = described_class.new(handle: mock_handle, credential_type: :pkcs12)

      hash_set = { creds1 => 'value1' }
      expect(hash_set[creds2]).to eq('value1')
    end
  end
end
