# frozen_string_literal: true

# Smoke spec — proves the gem loads + the cdylib resolves, against
# a real libpdf_oxide.{so,dylib,dll}.  Mirrors the Java JNI smoke
# coverage in `java/src/test/java/.../PdfDocumentTest.java`.

require 'spec_helper'
require 'fileutils'
require 'tmpdir'

RSpec.describe 'libpdf_oxide cdylib smoke' do
  it 'loads the gem with the expected version' do
    expect(defined?(PdfOxide)).to eq('constant')
    expect(PdfOxide::VERSION).to eq('0.3.60')
  end

  it 'exposes every public-API class' do
    %i[PdfDocument PdfPage Pdf DocumentEditor AutoExtractor
       MarkdownConverter PdfValidator PdfSigner PdfPolicy].each do |sym|
      expect(PdfOxide.const_defined?(sym)).to be(true), "PdfOxide::#{sym} not defined"
    end
  end

  it 'exposes the FFI Bindings module with the canonical cdylib symbols' do
    %i[pdf_from_markdown pdf_save pdf_save_to_bytes pdf_get_page_count pdf_free
       pdf_document_open pdf_document_free pdf_document_extract_text].each do |sym|
      expect(PdfOxide::Bindings).to respond_to(sym), "Bindings.#{sym} missing"
    end
  end

  it 'builds a real PDF from markdown via the cdylib' do
    bytes = PdfOxide::Pdf.from_markdown("# Hello\n\nworld.").to_bytes
    expect(bytes.bytesize).to be > 1024
    expect(bytes[0, 5]).to eq('%PDF-')
  end

  it 'opens a fixture PDF and reads page_count' do
    skip 'fixtures dir missing' unless Dir.exist?(PDF_OXIDE_FIXTURE_ROOT)

    PdfOxide::PdfDocument.open(File.join(PDF_OXIDE_FIXTURE_ROOT, 'simple.pdf')) do |doc|
      expect(doc.page_count).to be > 0
    end
  end
end
