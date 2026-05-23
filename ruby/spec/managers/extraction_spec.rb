# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Extraction do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe '#extract_text' do
    it 'extracts text from page' do
      allow(FFI::Bindings).to receive(:pdf_document_extract_text).and_return('Extracted text')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('Extracted text').and_return('Extracted text')

      result = manager.extract_text(0)
      expect(result).to eq('Extracted text')
    end

    it 'validates page index' do
      expect { manager.extract_text(-1) }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'handles nil text' do
      allow(FFI::Bindings).to receive(:pdf_document_extract_text).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:from_c_string).with(nil).and_return(nil)

      result = manager.extract_text(0)
      expect(result).to be_nil
    end
  end

  describe '#extract_text_all' do
    it 'extracts text from all pages' do
      allow(manager).to receive(:extract_text).and_return('Page text')

      result = manager.extract_text_all
      expect(result).to be_an(Array)
      expect(result.length).to eq(5)
    end
  end

  describe '#extract_to_markdown' do
    it 'extracts page text as markdown' do
      allow(FFI::Bindings).to receive(:pdf_document_to_markdown).and_return('# Markdown')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('# Markdown').and_return('# Markdown')

      result = manager.extract_to_markdown(0)
      expect(result).to include('Markdown')
    end

    it 'validates page index' do
      expect { manager.extract_to_markdown(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#extract_to_html' do
    it 'extracts page text as HTML' do
      allow(FFI::Bindings).to receive(:pdf_document_to_html).and_return('<html>...</html>')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('<html>...</html>').and_return('<html>...</html>')

      result = manager.extract_to_html(0)
      expect(result).to include('html')
    end
  end

  describe '#get_embedded_fonts' do
    it 'extracts embedded fonts from page' do
      allow(FFI::Bindings).to receive(:pdf_document_get_embedded_fonts).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_font_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_oxide_font_list_free)

      result = manager.get_embedded_fonts(0)
      expect(result).to be_an(Array)
      expect(result).to be_empty
    end

    it 'validates page index' do
      expect { manager.get_embedded_fonts(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#get_embedded_images' do
    it 'extracts embedded images from page' do
      allow(FFI::Bindings).to receive(:pdf_document_get_embedded_images).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_image_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_oxide_image_list_free)

      result = manager.get_embedded_images(0)
      expect(result).to be_an(Array)
      expect(result).to be_empty
    end
  end

  describe '#get_text_statistics' do
    it 'returns text statistics for page' do
      allow(FFI::Bindings).to receive(:pdf_document_get_text_statistics).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_text_stats_get_character_count).and_return(100)
      allow(FFI::Bindings).to receive(:pdf_oxide_text_stats_get_word_count).and_return(20)
      allow(FFI::Bindings).to receive(:pdf_oxide_text_stats_free)

      result = manager.get_text_statistics(0)
      expect(result).to be_a(Hash)
      expect(result).to have_key(:character_count)
    end
  end

  describe '#extract_image' do
    it 'extracts image to file' do
      allow(FFI::Bindings).to receive(:pdf_document_extract_image).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('output.png')

      result = manager.extract_image(0, 0, 'output.png')
      expect(result).to be true
    end

    it 'creates output directory if needed' do
      allow(FFI::Bindings).to receive(:pdf_document_extract_image).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('output.png')
      allow(Dir).to receive(:exist?).and_return(false)
      allow(FileUtils).to receive(:mkdir_p)

      manager.extract_image(0, 0, 'output.png')
      expect(FileUtils).to have_received(:mkdir_p)
    end
  end

  describe '#extract_all_images' do
    it 'extracts all images from page' do
      allow(manager).to receive(:get_embedded_images).and_return([
        double(index: 0),
        double(index: 1)
      ])
      allow(manager).to receive(:extract_image).and_return(true)

      result = manager.extract_all_images(0, 'output/')
      expect(result).to be true
    end
  end
end
