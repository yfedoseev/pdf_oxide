# frozen_string_literal: true

# Core functional test-parity suite (Ruby) — mirrors the shared cross-language
# spec (docs/releases/plans/v0.3.61/core-test-parity-spec.md) with the idiomatic
# Ruby API.

require 'pdf_oxide'

RSpec.describe 'core parity (Ruby)' do
  fixture = File.expand_path('../../tests/fixtures/simple.pdf', __dir__)

  it 'open + page count == 1' do
    PdfOxide::PdfDocument.open(fixture) do |doc|
      expect(doc.page_count).to eq(1)
    end
  end

  it 'extract text returns a String' do
    PdfOxide::PdfDocument.open(fixture) { |doc| expect(doc.extract_text(0)).to be_a(String) }
  end

  it 'convert markdown / html return Strings' do
    PdfOxide::PdfDocument.open(fixture) do |doc|
      expect(doc.to_markdown(0)).to be_a(String)
      expect(doc.to_html(0)).to be_a(String)
    end
  end

  it 'search returns an Array' do
    PdfOxide::PdfDocument.open(fixture) { |doc| expect(doc.search('the')).to be_a(Array) }
  end

  it 'structured extraction works' do
    PdfOxide::PdfDocument.open(fixture) { |doc| expect(doc.extract_structured(0)).not_to be_nil }
  end

  it 'create pdf from text → %PDF' do
    bytes = PdfOxide::Pdf.from_text('Core parity across all bindings.').to_bytes
    expect(bytes[0, 5]).to eq('%PDF-')
  end

  it 'opening a missing path raises' do
    expect { PdfOxide::PdfDocument.open('/no/such/file/does/not/exist.pdf') }.to raise_error(StandardError)
  end

  it 'exposes version 0.3.67' do
    expect(PdfOxide::VERSION).to eq('0.3.67')
  end
end
