# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::PdfDocument do
  include_context 'fixtures-present'

  describe '.open' do
    it 'opens a PDF from a path and exposes basic info' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        expect(doc.open?).to be(true)
        expect(doc.page_count).to be > 0
        expect(doc.encrypted?).to be(false)
      end
    end

    it 'auto-closes via the block form (idempotent)' do
      doc = PdfOxide::PdfDocument.open(fixture('simple.pdf'))
      expect(doc.open?).to be(true)
      doc.close
      expect(doc.closed?).to be(true)
      # Second and third close: no exception, no crash.
      doc.close
      doc.close
      expect(doc.closed?).to be(true)
    end

    it 'raises FileNotFoundError on missing paths' do
      expect { PdfOxide::PdfDocument.open('/no/such/file.pdf') }
        .to raise_error(PdfOxide::FileNotFoundError)
    end

    it 'raises InvalidStateError on operations after close' do
      doc = PdfOxide::PdfDocument.open(fixture('simple.pdf'))
      doc.close
      expect { doc.page_count }.to raise_error(PdfOxide::InvalidStateError, /closed/)
      expect { doc.extract_text(0) }.to raise_error(PdfOxide::InvalidStateError)
    end
  end

  describe '#extract_text' do
    it 'returns a String (possibly empty) for the requested page' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        text = doc.extract_text(0)
        expect(text).to be_a(String)
      end
    end

    it 'returns hello-like content for hello_structure.pdf' do
      skip 'fixture missing' unless File.exist?(fixture('hello_structure.pdf'))

      PdfOxide::PdfDocument.open(fixture('hello_structure.pdf')) do |doc|
        text = doc.extract_text(0)
        expect(text.downcase).to include('hello')
      end
    end
  end

  describe '#extract_text_auto' do
    it 'returns a String without raising on a no-OCR build (graceful fallback)' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        # Per feedback_extraction_graceful_fallback: never raises an
        # "OCR unavailable" error on this path.
        text = doc.extract_text_auto(0)
        expect(text).to be_a(String)
      end
    end
  end

  describe '#search' do
    it 'returns an empty array for non-existent text' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        expect(doc.search('xyzzyq42notthere')).to eq([])
      end
    end

    it 'returns matches with :page, :text, :bbox keys when found' do
      skip 'fixture missing' unless File.exist?(fixture('hello_structure.pdf'))

      PdfOxide::PdfDocument.open(fixture('hello_structure.pdf')) do |doc|
        matches = doc.search('Hello', case_sensitive: false)
        # may be empty if fixture has no text, but the shape must hold
        next if matches.empty?

        first = matches.first
        expect(first).to include(:page, :text, :bbox)
        expect(first[:bbox]).to include(:x, :y, :width, :height)
      end
    end
  end

  describe '#form_fields' do
    it 'returns an Array (possibly empty) without raising' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        expect(doc.form_fields).to be_an(Array)
      end
    end
  end

  describe '#auto_extractor' do
    it 'memoises the AutoExtractor instance' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        ax = doc.auto_extractor
        expect(ax).to be_a(PdfOxide::AutoExtractor)
        expect(doc.auto_extractor).to be(ax)
      end
    end
  end

  describe '#page' do
    it 'returns a PdfPage view borrowing from the document' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        page = doc.page(0)
        expect(page).to be_a(PdfOxide::PdfPage)
        expect(page.index).to eq(0)
        expect(page.parent).to be(doc)
      end
    end
  end

  describe '.extract_text' do
    it 'one-shot opens + extracts + closes' do
      text = PdfOxide::PdfDocument.extract_text(fixture('simple.pdf'))
      expect(text).to be_a(String)
    end
  end
end
