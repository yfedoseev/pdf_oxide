# frozen_string_literal: true

# Phase 3 acceptance test for split-by-bookmarks plan (v0.3.50).

require 'spec_helper'

PDF_OXIDE_FIXTURE_ROOT = File.expand_path('../../../tests/fixtures', __dir__) unless defined?(PDF_OXIDE_FIXTURE_ROOT)

RSpec.describe 'Outline#plan_split_by_bookmarks', :skip_mock do
  let(:outline_pdf) { File.join(PDF_OXIDE_FIXTURE_ROOT, 'outline.pdf') }

  it 'returns the decoded JSON plan as an Array of segment hashes' do
    doc  = PdfOxide::Document.open(outline_pdf)
    plan = doc.outline.plan_split_by_bookmarks

    expect(plan).to be_an(Array)
    expect(plan).not_to be_empty
    first = plan.first
    expect(first).to include('start_page', 'end_page')
    expect(first['start_page']).to be_a(Integer)
    expect(first['end_page']).to be_a(Integer)
    doc.close
  end

  it 'accepts an options hash and serializes it to JSON for the C call' do
    doc  = PdfOxide::Document.open(outline_pdf)
    # `level` is an upstream option; even when we pass an unknown
    # key the call must still return a plan (the C side ignores
    # unknown keys; the wire format is forgiving by design).
    plan = doc.outline.plan_split_by_bookmarks(level: 1)
    expect(plan).to be_an(Array)
    doc.close
  end
end
