# frozen_string_literal: true

# Phase 3 acceptance test for AutoExtractor (v0.3.51 #519).
#
# Exercises the typed-reason auto-extraction surface end-to-end:
# classify_page → extract_text → extract_page, plus the
# graceful-fallback contract (no opaque "OCR unavailable" error
# even when the build was compiled without the `ocr` feature —
# the reason :ocr_requested_but_unavailable surfaces instead).

require 'spec_helper'

PDF_OXIDE_FIXTURE_ROOT = File.expand_path('../../../tests/fixtures', __dir__) unless defined?(PDF_OXIDE_FIXTURE_ROOT)

RSpec.describe PdfOxide::AutoExtractor, :skip_mock do
  let(:simple_pdf) { File.join(PDF_OXIDE_FIXTURE_ROOT, 'simple.pdf') }
  let(:rich_pdf)   { File.join(PDF_OXIDE_FIXTURE_ROOT, '1.pdf') }

  it 'classifies a page and returns a typed reason' do
    doc = PdfOxide::Document.open(simple_pdf)
    ax  = PdfOxide::AutoExtractor.new(doc)
    result = ax.classify_page(0)

    expect(result).to be_a(PdfOxide::AutoExtractResult)
    expect(PdfOxide::ExtractReason::ALL).to include(result.reason)
    expect(PdfOxide::PageKind::ALL).to include(result.kind)
    expect(result.classification).to be_a(Hash)
    doc.close
  end

  it 'extracts text via the auto-router and surfaces a typed reason' do
    doc = PdfOxide::Document.open(rich_pdf)
    ax  = PdfOxide::AutoExtractor.new(doc)
    result = ax.extract_text(0)

    expect(result).to be_a(PdfOxide::AutoExtractResult)
    expect(PdfOxide::ExtractReason::ALL).to include(result.reason)
    expect(result.text).to be_a(String)
    # Native-text PDF — should NOT report OCR fallback.
    expect(result).not_to be_ocr_fallback
    doc.close
  end

  it 'returns a rich JSON envelope from extract_page' do
    doc = PdfOxide::Document.open(rich_pdf)
    ax  = PdfOxide::AutoExtractor.new(doc)
    result = ax.extract_page(0)

    expect(result.classification).to include('page', 'kind', 'text', 'reason', 'confidence')
    expect(result.confidence).to be >= 0.0
    expect(result.confidence).to be <= 1.0
    doc.close
  end

  it 'classifies the whole document' do
    doc = PdfOxide::Document.open(rich_pdf)
    ax  = PdfOxide::AutoExtractor.new(doc)
    cls = ax.classify_document

    expect(cls).to be_a(Hash)
    expect(cls).to include('pages')
    expect(cls['pages']).to be_an(Array)
    doc.close
  end

  it 'exposes the prefetch-availability gate (graceful-fallback)' do
    # Whether OCR is provisionable is a build-time property; either
    # outcome is valid. The contract is that this method ALWAYS
    # returns a Boolean without raising.
    expect([true, false]).to include(PdfOxide::AutoExtractor.prefetch_available?)
  end

  it 'reaches AutoExtractor via Document#auto_extractor accessor' do
    doc = PdfOxide::Document.open(simple_pdf)
    ax  = doc.auto_extractor
    expect(ax).to be_a(PdfOxide::AutoExtractor)
    expect(ax).to be(doc.auto_extractor) # memoised
    doc.close
  end
end
