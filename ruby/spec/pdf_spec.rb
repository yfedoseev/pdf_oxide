# frozen_string_literal: true

require 'spec_helper'
require 'tmpdir'

RSpec.describe PdfOxide::Pdf do
  describe '.from_markdown' do
    it 'produces a valid PDF' do
      bytes = PdfOxide::Pdf.from_markdown("# Hello\n\nworld.").to_bytes
      expect(bytes).to be_a(String)
      expect(bytes.bytesize).to be > 1024
      expect(bytes[0, 5]).to eq('%PDF-')
    end

    it 'rejects empty input' do
      expect { PdfOxide::Pdf.from_markdown(nil) }.to raise_error(PdfOxide::ArgumentError)
      expect { PdfOxide::Pdf.from_markdown('') }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'supports the block form (auto-close)' do
      out = nil
      PdfOxide::Pdf.from_markdown('# x') { |pdf| out = pdf.to_bytes }
      expect(out).to start_with('%PDF-')
    end
  end

  describe '.from_html' do
    it 'produces a valid PDF' do
      bytes = PdfOxide::Pdf.from_html('<h1>Hi</h1>').to_bytes
      expect(bytes[0, 5]).to eq('%PDF-')
    end
  end

  describe '#save' do
    it 'writes a PDF to disk that the OS can re-read' do
      Dir.mktmpdir do |tmp|
        out = File.join(tmp, 'roundtrip.pdf')
        PdfOxide::Pdf.from_markdown('# Round trip').save(out)
        expect(File).to exist(out)
        expect(File.size(out)).to be > 512
        File.open(out, 'rb') { |f| expect(f.read(5)).to eq('%PDF-') }
      end
    end
  end

  describe '#close' do
    it 'is idempotent' do
      pdf = PdfOxide::Pdf.from_markdown('# x')
      pdf.close
      expect(pdf.closed?).to be(true)
      pdf.close
      pdf.close
      expect(pdf.closed?).to be(true)
    end

    it 'raises on operations after close' do
      pdf = PdfOxide::Pdf.from_markdown('# x')
      pdf.close
      expect { pdf.to_bytes }.to raise_error(PdfOxide::InvalidStateError)
    end
  end

  describe '.version + .prefetch_available?' do
    it 'reports the library version' do
      expect(PdfOxide::Pdf.version).to eq(PdfOxide::VERSION)
    end

    it 'reports OCR availability as a Boolean (no raise)' do
      expect([true, false]).to include(PdfOxide::Pdf.prefetch_available?)
    end
  end
end
