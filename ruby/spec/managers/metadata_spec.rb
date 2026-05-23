# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Metadata do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  before do
    allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Test String')
    allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('Test String')
  end

  describe '#title' do
    it 'returns document title' do
      allow(FFI::Bindings).to receive(:pdf_document_get_title).and_return('Test Title')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('Test Title').and_return('Test Title')

      result = manager.title
      expect(result).to eq('Test Title')
    end

    it 'handles nil title' do
      allow(FFI::Bindings).to receive(:pdf_document_get_title).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:from_c_string).with(nil).and_return(nil)

      result = manager.title
      expect(result).to be_nil
    end
  end

  describe '#set_title' do
    it 'sets document title' do
      expect(FFI::Bindings).to receive(:pdf_document_set_title).and_return(true)

      result = manager.set_title('New Title')
      expect(result).to be true
    end

    it 'converts title to UTF-8' do
      allow(FFI::Bindings).to receive(:pdf_document_set_title)

      manager.set_title('New Title')
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('New Title')
    end
  end

  describe '#author' do
    it 'returns document author' do
      allow(FFI::Bindings).to receive(:pdf_document_get_author).and_return('John Doe')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('John Doe').and_return('John Doe')

      result = manager.author
      expect(result).to eq('John Doe')
    end
  end

  describe '#subject' do
    it 'returns document subject' do
      allow(FFI::Bindings).to receive(:pdf_document_get_subject).and_return('Test Subject')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('Test Subject').and_return('Test Subject')

      result = manager.subject
      expect(result).to eq('Test Subject')
    end
  end

  describe '#keywords' do
    it 'returns document keywords' do
      allow(FFI::Bindings).to receive(:pdf_document_get_keywords).and_return('keyword1, keyword2')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('keyword1, keyword2').and_return('keyword1, keyword2')

      result = manager.keywords
      expect(result).to eq('keyword1, keyword2')
    end
  end

  describe '#creation_date' do
    it 'returns creation date timestamp' do
      allow(FFI::Bindings).to receive(:pdf_document_get_creation_date).and_return(1234567890)

      result = manager.creation_date
      expect(result).to eq(1234567890)
    end
  end

  describe '#modification_date' do
    it 'returns modification date timestamp' do
      allow(FFI::Bindings).to receive(:pdf_document_get_modification_date).and_return(1234567890)

      result = manager.modification_date
      expect(result).to eq(1234567890)
    end
  end

  describe '#all' do
    it 'returns all metadata as hash' do
      allow(FFI::Bindings).to receive(:pdf_document_get_title).and_return('Test Title')
      allow(FFI::Bindings).to receive(:pdf_document_get_author).and_return('John Doe')
      allow(FFI::Bindings).to receive(:pdf_document_get_subject).and_return('Subject')
      allow(FFI::Bindings).to receive(:pdf_document_get_keywords).and_return('keywords')
      allow(FFI::Bindings).to receive(:pdf_document_get_creator).and_return('Creator')
      allow(FFI::Bindings).to receive(:pdf_document_get_producer).and_return('Producer')
      allow(FFI::Bindings).to receive(:pdf_document_get_creation_date).and_return(1234567890)
      allow(FFI::Bindings).to receive(:pdf_document_get_modification_date).and_return(1234567890)
      allow(FFI::StringMarshaller).to receive(:from_c_string) do |val|
        val.is_a?(String) ? val : nil
      end

      result = manager.all
      expect(result).to be_a(Hash)
      expect(result).to have_key(:title)
      expect(result).to have_key(:author)
    end
  end

  describe '#to_h' do
    it 'returns all metadata as hash' do
      allow(manager).to receive(:all).and_return({ title: 'Test', author: 'John' })

      result = manager.to_h
      expect(result).to eq({ title: 'Test', author: 'John' })
    end
  end

  describe '#check_document!' do
    it 'raises error when document is closed' do
      allow(mock_document).to receive(:closed?).and_return(true)

      expect { manager.title }.to raise_error(PdfOxide::StateError)
    end
  end
end
