# frozen_string_literal: true

# Phase 3 acceptance test for Models subsystem (v0.3.51 #519).
#
# Graceful-fallback contract: `Models.prefetch` NEVER raises a bare
# "OCR unavailable" error.  On builds without the `ocr` feature it
# still creates the cache directory (so callers can stage offline
# models) but `Models.available?` returns false so callers can
# branch.

require 'spec_helper'

RSpec.describe PdfOxide::Models, :skip_mock do
  it 'reports build-time OCR provisioning availability without raising' do
    expect([true, false]).to include(PdfOxide::Models.available?)
  end

  it 'returns a Hash from #manifest (may be empty on no-ocr builds)' do
    manifest = PdfOxide::Models.manifest
    expect(manifest).to be_a(Hash)
    # When OCR is compiled in the manifest lists at least one
    # language; otherwise it's the empty Hash.  Both are valid.
  end

  it 'prefetches without raising, returning a directory path' do
    # Graceful-fallback: even on no-ocr builds the cache dir is
    # created and its path is returned.  Empty string when the
    # build truly cannot stage models.
    path = PdfOxide::Models.prefetch(['eng'])
    expect(path).to be_a(String)
  end
end
