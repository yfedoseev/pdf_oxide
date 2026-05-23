# frozen_string_literal: true

# Phase 3 acceptance test for RedactionManager (v0.3.50 #231).
#
# Destructive redaction is a security operation per
# `feedback_extraction_graceful_fallback`: every non-zero return
# from the C ABI fails closed.  These tests verify the
# happy path and a basic security-op contract.

require 'spec_helper'

PDF_OXIDE_FIXTURE_ROOT = File.expand_path('../../../tests/fixtures', __dir__) unless defined?(PDF_OXIDE_FIXTURE_ROOT)

RSpec.describe PdfOxide::RedactionManager, :skip_mock do
  let(:source) { File.join(PDF_OXIDE_FIXTURE_ROOT, 'simple.pdf') }

  it 'adds, counts, applies, and serialises a redaction' do
    PdfOxide::RedactionManager.open(source) do |r|
      r.add(page: 0, rect: [100.0, 200.0, 300.0, 250.0])
      expect(r.count_for(0)).to eq(1)
      r.apply!

      bytes = r.to_bytes
      expect(bytes).to be_a(String)
      expect(bytes.bytesize).to be > 50
      expect(bytes[0, 5]).to eq('%PDF-')
    end
  end

  it 'refuses to_bytes / save before apply! (state guard)' do
    PdfOxide::RedactionManager.open(source) do |r|
      r.add(page: 0, rect: [100.0, 200.0, 300.0, 250.0])
      expect { r.to_bytes }.to raise_error(PdfOxide::StateError, /apply!/)
    end
  end

  it 'rejects malformed rects (argument guard)' do
    PdfOxide::RedactionManager.open(source) do |r|
      expect {
        r.add(page: 0, rect: [1.0, 2.0])
      }.to raise_error(PdfOxide::ArgumentError, /4 numeric/)
    end
  end

  it 'writes a destructively redacted file with save' do
    Dir.mktmpdir do |tmp|
      out_path = File.join(tmp, 'redacted.pdf')
      PdfOxide::RedactionManager.open(source) do |r|
        r.add(page: 0, rect: [100.0, 200.0, 300.0, 250.0])
        r.apply!
        r.save(out_path)
      end
      expect(File).to exist(out_path)
      expect(File.size(out_path)).to be > 50
      File.open(out_path, 'rb') { |f| expect(f.read(5)).to eq('%PDF-') }
    end
  end
end
