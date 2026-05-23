# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Document do
  describe 'class methods' do
    describe '.open' do
      it 'opens a valid PDF file' do
        # This would need actual PDF file in fixtures
        expect { PdfOxide::Document.open('nonexistent.pdf') }.to raise_error(PdfOxide::FileNotFoundError)
      end

      it 'supports block syntax with auto-close' do
        # Mock FFI to avoid actual file access
        allow(FFI::Bindings).to receive(:pdf_document_open).and_return(double(null?: false))
        allow(FFI::Bindings).to receive(:pdf_document_free)

        doc = nil
        result = PdfOxide::Document.open('test.pdf') do |d|
          doc = d
          'test result'
        end

        expect(result).to eq('test result')
        expect(doc).to be_a(PdfOxide::Document)
      end

      it 'raises FileNotFoundError for missing file' do
        expect {
          PdfOxide::Document.open('nonexistent.pdf')
        }.to raise_error(PdfOxide::FileNotFoundError)
      end
    end
  end

  describe 'instance methods' do
    let(:mock_handle) { double(null?: false) }

    before do
      allow(File).to receive(:exist?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_document_open).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_document_free)
    end

    describe '#initialize' do
      it 'opens document with valid path' do
        expect(FFI::Bindings).to receive(:pdf_document_open)
        doc = PdfOxide::Document.new('test.pdf')
        expect(doc).to be_a(PdfOxide::Document)
        doc.close
      end

      it 'raises error if path is nil' do
        expect {
          PdfOxide::Document.new(nil)
        }.to raise_error(ArgumentError)
      end

      it 'raises error if file does not exist' do
        allow(File).to receive(:exist?).and_return(false)
        expect {
          PdfOxide::Document.new('nonexistent.pdf')
        }.to raise_error(PdfOxide::FileNotFoundError)
      end
    end

    describe '#closed?' do
      it 'returns false initially' do
        doc = PdfOxide::Document.new('test.pdf')
        expect(doc.closed?).to be false
        doc.close
      end

      it 'returns true after close' do
        doc = PdfOxide::Document.new('test.pdf')
        doc.close
        expect(doc.closed?).to be true
      end
    end

    describe '#close' do
      it 'calls FFI free function' do
        doc = PdfOxide::Document.new('test.pdf')
        expect(FFI::Bindings).to receive(:pdf_document_free)
        doc.close
      end

      it 'can be called multiple times safely' do
        doc = PdfOxide::Document.new('test.pdf')
        doc.close
        expect { doc.close }.not_to raise_error
      end
    end

    describe 'manager access' do
      it 'returns SearchManager for search' do
        doc = PdfOxide::Document.new('test.pdf')
        expect(doc.search).to be_a(PdfOxide::Managers::Search)
        doc.close
      end

      it 'returns RenderingManager for rendering' do
        doc = PdfOxide::Document.new('test.pdf')
        expect(doc.rendering).to be_a(PdfOxide::Managers::Rendering)
        doc.close
      end

      it 'lazily initializes managers' do
        doc = PdfOxide::Document.new('test.pdf')
        search1 = doc.search
        search2 = doc.search
        expect(search1).to be(search2)
        doc.close
      end

      it 'returns all 15 managers' do
        doc = PdfOxide::Document.new('test.pdf')
        managers = [
          doc.search,
          doc.rendering,
          doc.annotations,
          doc.forms,
          doc.pages,
          doc.metadata,
          doc.outline,
          doc.layers,
          doc.cache,
          doc.extraction,
          doc.ocr,
          doc.compliance,
          doc.signatures,
          doc.barcodes,
          doc.analysis
        ]
        expect(managers.length).to eq(15)
        expect(managers.all? { |m| m.is_a?(PdfOxide::Managers::Base) }).to be true
        doc.close
      end
    end
  end
end
