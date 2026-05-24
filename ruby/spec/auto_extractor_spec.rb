# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::AutoExtractor do
  include_context 'fixtures-present'

  let(:simple_pdf) { fixture('simple.pdf') }

  describe '#classify_page' do
    it 'returns a typed reason + kind + confidence + classification hash' do
      PdfOxide::PdfDocument.open(simple_pdf) do |doc|
        ax = PdfOxide::AutoExtractor.new(doc)
        r = ax.classify_page(0)
        expect(PdfOxide::AutoExtractor::REASONS).to include(r[:reason])
        expect(PdfOxide::AutoExtractor::PAGE_KINDS).to include(r[:kind])
        expect(r[:confidence]).to be >= 0.0
        expect(r[:confidence]).to be <= 1.0
        expect(r[:classification]).to be_a(Hash)
      end
    end
  end

  describe '#extract_text' do
    it 'returns a Hash with :text and a typed :reason' do
      PdfOxide::PdfDocument.open(simple_pdf) do |doc|
        ax = PdfOxide::AutoExtractor.new(doc)
        r = ax.extract_text(0)
        expect(r[:text]).to be_a(String)
        expect(PdfOxide::AutoExtractor::REASONS).to include(r[:reason])
      end
    end
  end

  describe '#extract_page' do
    it 'returns a rich Hash with text + classification envelope' do
      PdfOxide::PdfDocument.open(simple_pdf) do |doc|
        ax = PdfOxide::AutoExtractor.new(doc)
        r = ax.extract_page(0)
        expect(r[:text]).to be_a(String)
        expect(r[:classification]).to be_a(Hash)
        expect(r[:classification]).to include('text', 'reason')
      end
    end
  end

  describe '#classify_document' do
    it 'returns a Hash (decoded JSON envelope)' do
      PdfOxide::PdfDocument.open(simple_pdf) do |doc|
        ax = PdfOxide::AutoExtractor.new(doc)
        cls = ax.classify_document
        expect(cls).to be_a(Hash)
      end
    end
  end

  describe '#ok? + #ocr_fallback?' do
    it 'classifies clean vs degraded reasons' do
      PdfOxide::PdfDocument.open(simple_pdf) do |doc|
        ax = PdfOxide::AutoExtractor.new(doc)
        expect(ax.ok?(:ok)).to be(true)
        expect(ax.ok?(:native_text_high_confidence)).to be(true)
        expect(ax.ok?(:ocr_requested_but_unavailable)).to be(false)
        expect(ax.ocr_fallback?(:ocr_requested_but_unavailable)).to be(true)
        expect(ax.ocr_fallback?(:ok)).to be(false)
      end
    end
  end

  describe '.prefetch_available?' do
    it 'returns a Boolean without raising (graceful-fallback)' do
      expect([true, false]).to include(PdfOxide::AutoExtractor.prefetch_available?)
    end
  end
end
