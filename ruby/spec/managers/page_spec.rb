# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Page do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe '#count' do
    it 'returns page count from document' do
      result = manager.count
      expect(result).to eq(5)
    end
  end

  describe '#exists' do
    it 'returns true for valid page index' do
      expect(manager.exists(0)).to be true
      expect(manager.exists(4)).to be true
    end

    it 'returns false for invalid page index' do
      expect(manager.exists(-1)).to be false
      expect(manager.exists(5)).to be false
    end
  end

  describe '#get_dimensions' do
    it 'returns page dimensions' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_width).and_return(612.0)
      allow(FFI::Bindings).to receive(:pdf_document_get_page_height).and_return(792.0)

      result = manager.get_dimensions(0)
      expect(result).to be_a(PdfOxide::Types::PageDimensions)
      expect(result.width).to eq(612.0)
      expect(result.height).to eq(792.0)
    end

    it 'validates page index' do
      expect { manager.get_dimensions(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#get_page_width' do
    it 'returns page width' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_width).and_return(612.0)

      result = manager.get_page_width(0)
      expect(result).to eq(612.0)
    end
  end

  describe '#get_page_height' do
    it 'returns page height' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_height).and_return(792.0)

      result = manager.get_page_height(0)
      expect(result).to eq(792.0)
    end
  end

  describe '#get_aspect_ratio' do
    it 'calculates aspect ratio from dimensions' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_width).and_return(612.0)
      allow(FFI::Bindings).to receive(:pdf_document_get_page_height).and_return(792.0)

      result = manager.get_aspect_ratio(0)
      expect(result).to be_close(0.7727, 0.001)
    end
  end

  describe '#rotate_page' do
    it 'rotates page by specified degrees' do
      expect(FFI::Bindings).to receive(:pdf_document_rotate_page).and_return(true)

      result = manager.rotate_page(0, 90)
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.rotate_page(-1, 90) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#delete_page' do
    it 'deletes page from document' do
      expect(FFI::Bindings).to receive(:pdf_document_delete_page).and_return(true)

      result = manager.delete_page(0)
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.delete_page(5) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#insert_page' do
    it 'inserts page at specified index' do
      expect(FFI::Bindings).to receive(:pdf_document_insert_page).and_return(true)

      result = manager.insert_page(2)
      expect(result).to be true
    end
  end

  describe '#duplicate_page' do
    it 'duplicates page' do
      expect(FFI::Bindings).to receive(:pdf_document_duplicate_page).and_return(true)

      result = manager.duplicate_page(0)
      expect(result).to be true
    end
  end

  describe '#get_page_rotation' do
    it 'returns page rotation in degrees' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_rotation).and_return(90)

      result = manager.get_page_rotation(0)
      expect(result).to eq(90)
    end
  end
end
