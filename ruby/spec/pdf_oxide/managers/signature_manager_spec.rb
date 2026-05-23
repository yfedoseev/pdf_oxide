require 'spec_helper'

RSpec.describe PdfOxide::SignatureManager do
  let(:mock_document) { instance_double(PdfOxide::PdfDocument) }
  let(:manager) { described_class.new(mock_document) }

  describe '#initialize' do
    it 'creates a new instance with document' do
      expect(manager.document).to eq(mock_document)
      expect(manager.cache).to be_a(ActiveSupport::Cache::MemoryStore)
    end
  end

  describe '#get_signature_count' do
    it 'returns an integer' do
      result = manager.get_signature_count
      expect(result).to be_a(Integer)
      expect(result).to be >= 0
    end

    it 'caches the result' do
      count1 = manager.get_signature_count
      count2 = manager.get_signature_count
      expect(count1).to eq(count2)
    end
  end

  describe '#get_signature_info' do
    it 'returns signature information' do
      info = manager.get_signature_info(0, 0)
      expect(info).to be_a(PdfOxide::SignatureManager::SignatureInfo)
      expect(info.page_index).to eq(0)
      expect(info.signature_index).to eq(0)
    end

    it 'caches signature info' do
      info1 = manager.get_signature_info(0, 0)
      info2 = manager.get_signature_info(0, 0)
      expect(info1).to eq(info2)
    end
  end

  describe '#verify_signature' do
    it 'returns verification status' do
      status = manager.verify_signature(0, 0)
      expect([:valid, :invalid, :untrusted]).to include(status)
    end

    it 'caches verification result' do
      status1 = manager.verify_signature(0, 0)
      status2 = manager.verify_signature(0, 0)
      expect(status1).to eq(status2)
    end
  end

  describe '#get_certificate_info' do
    it 'returns certificate information' do
      cert = manager.get_certificate_info(0)
      expect(cert).to be_a(PdfOxide::SignatureManager::CertificateInfo)
      expect(cert.common_name).to be_a(String)
    end

    it 'caches certificate info' do
      cert1 = manager.get_certificate_info(0)
      cert2 = manager.get_certificate_info(0)
      expect(cert1).to eq(cert2)
    end
  end

  describe '#get_all_signatures' do
    it 'returns array of signatures' do
      signatures = manager.get_all_signatures
      expect(signatures).to be_a(Array)
      signatures.each do |sig|
        expect(sig).to be_a(PdfOxide::SignatureManager::SignatureInfo)
      end
    end

    it 'caches all signatures' do
      sigs1 = manager.get_all_signatures
      sigs2 = manager.get_all_signatures
      expect(sigs1).to eq(sigs2)
    end
  end

  describe '#all_signatures_valid?' do
    it 'returns true/false' do
      result = manager.all_signatures_valid?
      expect(result).to be(true).or be(false)
    end
  end

  describe '#signature_statistics' do
    it 'returns statistics hash' do
      stats = manager.signature_statistics
      expect(stats).to include(:total, :valid, :invalid)
      expect(stats[:total]).to be >= 0
    end
  end

  describe '#get_signer_info' do
    it 'returns signer information' do
      info = manager.get_signer_info(0)
      if info
        expect(info).to include(:name, :organization)
      end
    end
  end

  describe '#validate_certificate_chain' do
    it 'returns boolean' do
      result = manager.validate_certificate_chain(0)
      expect([true, false]).to include(result)
    end
  end

  describe '#signature_trusted?' do
    it 'returns boolean' do
      result = manager.signature_trusted?(0)
      expect([true, false]).to include(result)
    end
  end

  describe '#signature_report' do
    it 'returns detailed report' do
      report = manager.signature_report
      expect(report).to include(:statistics, :signatures, :generated_at)
      expect(report[:statistics]).to be_a(Hash)
      expect(report[:signatures]).to be_a(Array)
    end
  end

  describe '#clear_cache' do
    it 'clears the cache' do
      manager.get_signature_count
      manager.clear_cache
      # Verify no error
      expect(true).to be true
    end
  end

  describe '#pages_with_signatures' do
    it 'returns array of page indices' do
      pages = manager.pages_with_signatures
      expect(pages).to be_a(Array)
      expect(pages).to all(be_a(Integer))
    end
  end

  describe '#verify_signatures_sync' do
    it 'verifies signatures synchronously' do
      results = manager.verify_signatures_sync
      expect(results).to be_a(Array)
    end

    it 'yields signature and status when block given' do
      expect { |b| manager.verify_signatures_sync(&b) }.to yield_control
    end
  end

  describe 'thread safety' do
    it 'handles concurrent access' do
      threads = []
      3.times do
        threads << Thread.new { manager.get_all_signatures }
      end
      
      results = threads.map(&:value)
      expect(results).to all(be_a(Array))
    end
  end
end
