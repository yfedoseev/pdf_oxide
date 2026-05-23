# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Annotation do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'annotation type constants' do
    it 'defines all annotation types' do
      expect(described_class::ANNOTATION_TYPE_TEXT).to eq(0)
      expect(described_class::ANNOTATION_TYPE_HIGHLIGHT).to eq(1)
      expect(described_class::ANNOTATION_TYPE_UNDERLINE).to eq(2)
      expect(described_class::ANNOTATION_TYPE_STRIKEOUT).to eq(3)
    end
  end

  describe '#annotation_count' do
    it 'returns number of annotations on page' do
      allow(FFI::Bindings).to receive(:pdf_document_get_annotation_count).and_return(3)

      result = manager.annotation_count(0)
      expect(result).to eq(3)
    end

    it 'validates page index' do
      expect { manager.annotation_count(-1) }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates page index is within bounds' do
      expect { manager.annotation_count(5) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#has_annotations?' do
    it 'returns true when page has annotations' do
      allow(manager).to receive(:annotation_count).and_return(3)

      result = manager.has_annotations?(0)
      expect(result).to be true
    end

    it 'returns false when page has no annotations' do
      allow(manager).to receive(:annotation_count).and_return(0)

      result = manager.has_annotations?(0)
      expect(result).to be false
    end
  end

  describe '#add_highlight' do
    it 'adds highlight annotation to page' do
      expect(FFI::Bindings).to receive(:pdf_document_add_highlight).and_return(true)

      result = manager.add_highlight(0, 10.0, 20.0, 100.0, 50.0)
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.add_highlight(-1, 10, 20, 100, 50) }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'converts coordinates to floats' do
      allow(FFI::Bindings).to receive(:pdf_document_add_highlight)

      manager.add_highlight(0, 10, 20, 100, 50)
      expect(FFI::Bindings).to have_received(:pdf_document_add_highlight)
        .with(mock_handle, 0, 10.0, 20.0, 100.0, 50.0, anything)
    end
  end

  describe '#add_underline' do
    it 'adds underline annotation to page' do
      expect(FFI::Bindings).to receive(:pdf_document_add_underline).and_return(true)

      result = manager.add_underline(0, 10.0, 20.0, 100.0, 50.0)
      expect(result).to be true
    end
  end

  describe '#add_strikeout' do
    it 'adds strikeout annotation to page' do
      expect(FFI::Bindings).to receive(:pdf_document_add_strikeout).and_return(true)

      result = manager.add_strikeout(0, 10.0, 20.0, 100.0, 50.0)
      expect(result).to be true
    end
  end

  describe '#add_comment' do
    it 'adds text annotation with comment' do
      expect(FFI::Bindings).to receive(:pdf_document_add_text_annotation).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('Comment text')

      result = manager.add_comment(0, 10.0, 20.0, 'Comment text')
      expect(result).to be true
    end

    it 'marshals comment text to UTF-8' do
      allow(FFI::Bindings).to receive(:pdf_document_add_text_annotation)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_call_original

      manager.add_comment(0, 10.0, 20.0, 'Comment')
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('Comment')
    end
  end

  describe '#delete_annotation' do
    it 'deletes annotation from page' do
      allow(manager).to receive(:annotation_count).and_return(1)
      expect(FFI::Bindings).to receive(:pdf_document_delete_annotation).and_return(true)

      result = manager.delete_annotation(0, 0)
      expect(result).to be true
    end

    it 'validates annotation index' do
      allow(manager).to receive(:annotation_count).and_return(0)

      expect { manager.delete_annotation(0, 0) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#flatten_annotations' do
    it 'flattens annotations on specific page' do
      expect(FFI::Bindings).to receive(:pdf_document_flatten_annotations).and_return(true)

      result = manager.flatten_annotations(0)
      expect(result).to be true
    end

    it 'flattens annotations on all pages when page_index is nil' do
      expect(FFI::Bindings).to receive(:pdf_document_flatten_annotations).at_least(:once).and_return(true)

      result = manager.flatten_annotations(nil)
      expect(result).to be true
    end
  end

  describe '#annotation_statistics' do
    it 'returns statistics about annotations' do
      allow(manager).to receive(:get_all_annotations).and_return([
        double(type: 1, page: 0),
        double(type: 2, page: 0),
        double(type: 1, page: 1)
      ])

      result = manager.annotation_statistics
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_annotations)
      expect(result[:total_annotations]).to eq(3)
    end
  end

  describe '#get_all_annotations' do
    it 'returns array of annotations from all pages' do
      allow(manager).to receive(:list_annotations).and_return([
        double(page: 0),
        double(page: 1)
      ])

      result = manager.get_all_annotations
      expect(result).to be_a(Array)
    end
  end

  describe '#import_annotations' do
    let(:json_annotations) do
      '[{"type":"highlight","page":0,"bbox":{"x":100,"y":200,"width":50,"height":20}}]'
    end

    let(:xfdf_annotations) do
      '<?xml version="1.0"?><xfdf><annots><highlight page="0" rect="100,200,150,220"/></annots></xfdf>'
    end

    it 'imports annotations from JSON string' do
      allow(manager).to receive(:add_highlight).and_return(true)

      result = manager.import_annotations(json_annotations)
      expect(result).to eq(1)
      expect(manager).to have_received(:add_highlight).with(0, 100, 200, 50, 20, {})
    end

    it 'imports annotations from XFDF string' do
      allow(manager).to receive(:add_highlight).and_return(true)

      result = manager.import_annotations(xfdf_annotations)
      expect(result).to eq(1)
    end

    it 'handles multiple annotation types' do
      json_multi = '[{"type":"highlight","page":0,"bbox":{"x":100,"y":200,"width":50,"height":20}},{"type":"underline","page":1,"bbox":{"x":150,"y":250,"width":60,"height":15}}]'
      allow(manager).to receive(:add_highlight).and_return(true)
      allow(manager).to receive(:add_underline).and_return(true)

      result = manager.import_annotations(json_multi)
      expect(result).to eq(2)
    end

    it 'raises error for empty input' do
      expect { manager.import_annotations('') }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error for invalid JSON' do
      expect { manager.import_annotations('not json') }.to raise_error(PdfOxide::ParseError)
    end

    it 'skips unsupported annotation types gracefully' do
      json_unknown = '[{"type":"unknown_type","page":0,"bbox":{"x":100,"y":200,"width":50,"height":20}}]'

      result = manager.import_annotations(json_unknown)
      expect(result).to eq(0)
    end
  end

  describe '#import_annotations_dsl' do
    it 'allows DSL-style annotation import' do
      allow(manager).to receive(:add_highlight).and_return(true)
      allow(manager).to receive(:add_comment).and_return(true)

      result = manager.import_annotations_dsl do |builder|
        builder.highlight(page: 0, x: 100, y: 200, width: 50, height: 20)
        builder.comment(page: 1, x: 150, y: 300, text: 'Test comment')
      end

      expect(result).to eq(2)
    end
  end
end

RSpec.describe PdfOxide::Managers::AnnotationBuilder do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { PdfOxide::Managers::Annotation.new(mock_document) }
  let(:builder) { described_class.new(manager) }

  describe '#highlight' do
    it 'adds highlight annotation via manager' do
      expect(manager).to receive(:add_highlight).with(0, 100, 200, 50, 20, {})

      builder.highlight(page: 0, x: 100, y: 200, width: 50, height: 20)
      expect(builder.import_count).to eq(1)
    end

    it 'returns self for chaining' do
      allow(manager).to receive(:add_highlight)

      result = builder.highlight(page: 0, x: 100, y: 200, width: 50, height: 20)
      expect(result).to be(builder)
    end
  end

  describe '#underline' do
    it 'adds underline annotation via manager' do
      expect(manager).to receive(:add_underline).with(1, 150, 250, 60, 15, {})

      builder.underline(page: 1, x: 150, y: 250, width: 60, height: 15)
      expect(builder.import_count).to eq(1)
    end
  end

  describe '#strikeout' do
    it 'adds strikeout annotation via manager' do
      expect(manager).to receive(:add_strikeout).with(2, 200, 300, 70, 10, {})

      builder.strikeout(page: 2, x: 200, y: 300, width: 70, height: 10)
      expect(builder.import_count).to eq(1)
    end
  end

  describe '#comment' do
    it 'adds text annotation via manager' do
      expect(manager).to receive(:add_comment).with(0, 100, 200, 'Test', {})

      builder.comment(page: 0, x: 100, y: 200, text: 'Test')
      expect(builder.import_count).to eq(1)
    end
  end

  describe 'aliases' do
    it 'has text as alias for comment' do
      expect(builder.method(:text)).to eq(builder.method(:comment))
    end

    it 'has note as alias for comment' do
      expect(builder.method(:note)).to eq(builder.method(:comment))
    end
  end

  describe 'chaining' do
    it 'supports method chaining' do
      allow(manager).to receive(:add_highlight)
      allow(manager).to receive(:add_underline)
      allow(manager).to receive(:add_comment)

      builder
        .highlight(page: 0, x: 100, y: 200, width: 50, height: 20)
        .underline(page: 0, x: 100, y: 250, width: 50, height: 10)
        .comment(page: 0, x: 100, y: 300, text: 'Note')

      expect(builder.import_count).to eq(3)
    end
  end
end
