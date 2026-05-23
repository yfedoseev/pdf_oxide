# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Signature do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'signature status constants' do
    it 'defines all signature statuses' do
      expect(described_class::SIGNATURE_STATUS_VALID).to eq(1)
      expect(described_class::SIGNATURE_STATUS_INVALID).to eq(2)
      expect(described_class::SIGNATURE_STATUS_UNTRUSTED).to eq(3)
      expect(described_class::SIGNATURE_STATUS_SELF_SIGNED).to eq(4)
    end
  end

  describe '#signature_count' do
    it 'returns number of signatures' do
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_count).and_return(2)

      result = manager.signature_count
      expect(result).to eq(2)
    end

    it 'returns zero when no signatures' do
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_count).and_return(0)

      result = manager.signature_count
      expect(result).to eq(0)
    end
  end

  describe '#has_signatures?' do
    it 'returns true when document has signatures' do
      allow(manager).to receive(:signature_count).and_return(1)

      result = manager.has_signatures?
      expect(result).to be true
    end

    it 'returns false when no signatures' do
      allow(manager).to receive(:signature_count).and_return(0)

      result = manager.has_signatures?
      expect(result).to be false
    end
  end

  describe '#get_signature' do
    it 'returns signature information' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_get_signer).and_return('John Doe')
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_get_timestamp).and_return(1234567890)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_get_status).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_get_reason).and_return('Signing')
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_get_location).and_return('New York')
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Value')

      result = manager.get_signature(0)
      expect(result).to be_a(Hash)
      expect(result[:index]).to eq(0)
    end

    it 'validates signature index' do
      allow(manager).to receive(:signature_count).and_return(0)

      expect { manager.get_signature(0) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#verify_signature' do
    it 'verifies signature' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_verify_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_verification_is_valid).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_oxide_verification_is_trusted).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_oxide_verification_is_self_signed).and_return(false)
      allow(FFI::Bindings).to receive(:pdf_oxide_verification_get_error).and_return('')
      allow(FFI::Bindings).to receive(:pdf_oxide_verification_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('')

      result = manager.verify_signature(0)
      expect(result).to be_a(Hash)
      expect(result[:valid]).to be true
    end
  end

  describe '#get_signer' do
    it 'returns signer name' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_signer).and_return('John Doe')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('John Doe').and_return('John Doe')

      result = manager.get_signer(0)
      expect(result).to eq('John Doe')
    end

    it 'returns Unknown for nil signer' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_signer).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:from_c_string).with(nil).and_return(nil)

      result = manager.get_signer(0)
      expect(result).to eq('Unknown')
    end
  end

  describe '#get_timestamp' do
    it 'returns signature timestamp' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_timestamp).and_return(1234567890)

      result = manager.get_timestamp(0)
      expect(result).to eq(1234567890)
    end
  end

  describe '#get_signature_status' do
    it 'returns signature status' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_status).and_return(1)

      result = manager.get_signature_status(0)
      expect(result).to eq(:valid)
    end

    it 'returns unknown for invalid status code' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature_status).and_return(99)

      result = manager.get_signature_status(0)
      expect(result).to eq(:unknown)
    end
  end

  describe '#is_signature_valid?' do
    it 'returns true for valid signature' do
      allow(manager).to receive(:get_signature_status).and_return(:valid)

      result = manager.is_signature_valid?(0)
      expect(result).to be true
    end

    it 'returns false for invalid signature' do
      allow(manager).to receive(:get_signature_status).and_return(:invalid)

      result = manager.is_signature_valid?(0)
      expect(result).to be false
    end
  end

  describe '#list_signatures' do
    it 'returns list of signatures with basic info' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(manager).to receive(:get_signer).and_return('John Doe')
      allow(manager).to receive(:get_signature_status).and_return(:valid)
      allow(manager).to receive(:get_timestamp).and_return(1234567890)

      result = manager.list_signatures
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:signer)
      expect(result.first).to have_key(:status)
    end
  end

  describe '#signature_statistics' do
    it 'returns signature statistics' do
      allow(manager).to receive(:list_signatures).and_return([
        { status: :valid },
        { status: :invalid }
      ])

      result = manager.signature_statistics
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_signatures)
      expect(result[:total_signatures]).to eq(2)
    end
  end

  describe '#sign_with_appearance' do
    let(:mock_credentials) { instance_double(PdfOxide::Types::SigningCredentials, handle: mock_handle) }

    it 'signs document with visual appearance' do
      allow(FFI::Bindings).to receive(:pdf_document_sign_with_appearance).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('')

      position = { x: 100, y: 200, width: 150, height: 50 }
      result = manager.sign_with_appearance(0, position, mock_credentials, reason: 'Approval')

      expect(result).to be true
      expect(FFI::Bindings).to have_received(:pdf_document_sign_with_appearance)
    end

    it 'validates page index' do
      position = { x: 100, y: 200, width: 150, height: 50 }
      expect { manager.sign_with_appearance(-1, position, mock_credentials) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates credentials type' do
      position = { x: 100, y: 200, width: 150, height: 50 }
      expect { manager.sign_with_appearance(0, position, 'invalid') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates position hash structure' do
      expect { manager.sign_with_appearance(0, { x: 100 }, mock_credentials) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'applies default algorithm (SHA256)' do
      allow(FFI::Bindings).to receive(:pdf_document_sign_with_appearance).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('')

      position = { x: 100, y: 200, width: 150, height: 50 }
      manager.sign_with_appearance(0, position, mock_credentials)

      expect(FFI::Bindings).to have_received(:pdf_document_sign_with_appearance) do |*args|
        expect(args[9]).to eq(1) # SHA-256 algorithm
      end
    end
  end

  describe '#add_timestamp' do
    it 'adds RFC 3161 timestamp to signature' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_add_timestamp).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('tsa_url')

      result = manager.add_timestamp(0, 'http://tsa.example.com')
      expect(result).to be true
    end

    it 'validates signature index' do
      allow(manager).to receive(:signature_count).and_return(0)
      expect { manager.add_timestamp(0, 'http://tsa.example.com') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates TSA URL is a string' do
      allow(manager).to receive(:signature_count).and_return(1)
      expect { manager.add_timestamp(0, 123) }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#co_sign' do
    let(:mock_credentials) { instance_double(PdfOxide::Types::SigningCredentials, handle: mock_handle) }

    it 'adds a co-signature to document' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_co_sign).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('reason')

      result = manager.co_sign(mock_credentials, 'Approval')
      expect(result).to eq(0) # Index of new signature
    end

    it 'validates credentials type' do
      expect { manager.co_sign('invalid', 'reason') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates reason is a string' do
      expect { manager.co_sign(mock_credentials, 123) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'uses empty string for default reason' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_co_sign).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('')

      manager.co_sign(mock_credentials)
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('')
    end
  end

  describe '#verify_all_signatures' do
    it 'verifies all signatures in document' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(manager).to receive(:is_signature_valid?).and_return(true)
      allow(manager).to receive(:get_signature_status).and_return(:valid)
      allow(manager).to receive(:get_signer).and_return('John Doe')
      allow(manager).to receive(:get_timestamp).and_return(1234567890)
      allow(FFI::Bindings).to receive(:pdf_verify_all_signatures).and_return(mock_handle)

      result = manager.verify_all_signatures
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:signature_index)
      expect(result.first).to have_key(:valid)
    end

    it 'validates trusted_certs is an array' do
      expect { manager.verify_all_signatures('invalid') }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#has_timestamp?' do
    it 'returns true when signature has timestamp' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_has_timestamp).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)

      result = manager.has_timestamp?(0)
      expect(result).to be true
    end

    it 'returns false when signature has no timestamp' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_has_timestamp).and_return(false)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)

      result = manager.has_timestamp?(0)
      expect(result).to be false
    end

    it 'validates signature index' do
      allow(manager).to receive(:signature_count).and_return(0)
      expect { manager.has_timestamp?(0) }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#get_signature_algorithm' do
    it 'returns digest algorithm' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_get_digest_algorithm).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)

      result = manager.get_signature_algorithm(0)
      expect(result).to eq(:sha256)
    end

    it 'returns unknown for invalid algorithm' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_get_digest_algorithm).and_return(99)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)

      result = manager.get_signature_algorithm(0)
      expect(result).to eq(:unknown)
    end
  end

  describe '#export_signature_json' do
    it 'exports signature as JSON' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_to_json).and_return('{"signer": "John Doe"}')
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('{"signer": "John Doe"}')

      result = manager.export_signature_json(0)
      expect(result).to be_a(Hash)
      expect(result['signer']).to eq('John Doe')
    end

    it 'returns empty hash on error' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_document_get_signature).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_signature_to_json).and_raise(StandardError)
      allow(FFI::Bindings).to receive(:pdf_oxide_signature_free)

      result = manager.export_signature_json(0)
      expect(result).to have_key(:error)
    end
  end

  describe '#remove_signature' do
    it 'removes signature from document' do
      allow(manager).to receive(:signature_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_remove_signature).and_return(true)

      result = manager.remove_signature(0)
      expect(result).to be true
    end

    it 'validates signature index' do
      allow(manager).to receive(:signature_count).and_return(0)
      expect { manager.remove_signature(0) }
        .to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#clear_all_signatures' do
    it 'clears all signatures from document' do
      allow(manager).to receive(:signature_count).and_return(2)
      allow(FFI::Bindings).to receive(:pdf_clear_all_signatures).and_return(true)

      result = manager.clear_all_signatures
      expect(result).to eq(2)
    end
  end
end
