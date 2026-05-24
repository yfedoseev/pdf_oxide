# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::PdfSigner do
  describe 'LEVELS' do
    it 'declares the canonical PAdES level codes' do
      expect(PdfOxide::PdfSigner::LEVELS).to eq(b: 0, t: 1, lt: 2, lta: 3)
    end
  end

  describe 'PadesSignOptions struct layout' do
    it 'matches PadesSignOptionsC (14 fields × 8 bytes on x86_64)' do
      expect(PdfOxide::PdfSigner::PadesSignOptions.size).to eq(14 * 8)
    end
  end

  describe 'security-op argument guards (fail-closed)' do
    it 'rejects a nil certificate handle' do
      expect do
        PdfOxide::PdfSigner.sign(pdf: '%PDF-1.7', certificate_handle: nil, level: :b)
      end.to raise_error(PdfOxide::ArgumentError, /certificate_handle/)
    end

    it 'rejects unknown PAdES levels' do
      fake = ::FFI::Pointer.new(0xdeadbeef)
      expect do
        PdfOxide::PdfSigner.sign(pdf: '%PDF-1.7', certificate_handle: fake, level: :forged)
      end.to raise_error(PdfOxide::ArgumentError, /level must be one of/)
    end

    it 'rejects empty pdf bytes' do
      fake = ::FFI::Pointer.new(0xdeadbeef)
      expect do
        PdfOxide::PdfSigner.sign(pdf: '', certificate_handle: fake, level: :b)
      end.to raise_error(PdfOxide::ArgumentError, /pdf/)
    end

    it 'requires tsa_url for B-T / B-LT / B-LTA' do
      fake = ::FFI::Pointer.new(0xdeadbeef)
      expect do
        PdfOxide::PdfSigner.sign(pdf: '%PDF-1.7', certificate_handle: fake, level: :t)
      end.to raise_error(PdfOxide::ArgumentError, /tsa_url/)
    end
  end
end
