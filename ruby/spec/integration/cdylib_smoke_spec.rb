# frozen_string_literal: true

# Phase 2 acceptance test.
#
# Loads the gem against the real libpdf_oxide cdylib and exercises one round-
# trip through the C ABI.  Unlike the 664 legacy mock-based examples (which
# never touch the native library and therefore failed to catch the 443
# phantom FFI declarations in the v0.3.47-era snapshot), this spec proves:
#
#   1. The bindings file resolves every `attach_function` against an actual
#      symbol in libpdf_oxide.so.
#   2. `PdfOxide::Creator` is not the empty-file stub it used to be — calling
#      `to_bytes` returns a real PDF (`%PDF-…` signature).
#   3. `pdf_save` round-trips the bytes to disk producing a valid file.
#
# Run with `LD_LIBRARY_PATH=…/target/release bundle exec rspec
# spec/integration/cdylib_smoke_spec.rb` (or set the path appropriately for
# macOS / Windows).

require 'spec_helper'
require 'fileutils'
require 'tmpdir'

RSpec.describe 'libpdf_oxide cdylib smoke', :skip_mock do
  it 'loads the gem without raising FFI::NotFoundError' do
    expect(defined?(PdfOxide)).to eq('constant')
    expect(PdfOxide::VERSION).to eq('0.3.55')
  end

  it 'exposes the FFI Bindings module with real cdylib symbols attached' do
    expect(PdfOxide::FFI::Bindings).to respond_to(:pdf_from_markdown)
    expect(PdfOxide::FFI::Bindings).to respond_to(:pdf_save)
    expect(PdfOxide::FFI::Bindings).to respond_to(:pdf_save_to_bytes)
    expect(PdfOxide::FFI::Bindings).to respond_to(:pdf_get_page_count)
    expect(PdfOxide::FFI::Bindings).to respond_to(:pdf_free)
  end

  it 'wires the surviving manager files into the load path' do
    # Some managers live under PdfOxide::Managers; the historically-
    # unloaded set lives at PdfOxide directly with a `Manager` suffix.
    # Phase 4 retired EditingManager, OptimizationManager and the legacy
    # PdfOxide::SignatureManager — all three referenced C symbols absent
    # from the current cdylib header.  RedactionManager + PadesSigner
    # (Phase 3) are the real replacements.
    expected = {
      'PdfOxide::Managers' => %i[
        Analysis Annotation Barcode Base Cache Certificate Compliance
        Extraction ExtractionStrategy Form Layer Metadata MetaManager Ocr
        Outline Page Rendering Search Signature Xfa
      ],
      'PdfOxide' => %i[
        AccessibilityManager EnterpriseManager
        RedactionManager PadesSigner
      ]
    }
    expected.each do |ns, classes|
      mod = ns.split('::').inject(Object) { |o, n| o.const_get(n) }
      classes.each do |c|
        expect(mod.const_defined?(c)).to be(true),
                                         "expected #{ns}::#{c} to be defined"
      end
    end
  end

  it 'builds a real PDF from markdown via the cdylib' do
    bytes = PdfOxide::Creator.from_markdown("# Hello\n\nworld.").to_bytes
    expect(bytes).to be_a(String)
    expect(bytes.bytesize).to be > 1024
    expect(bytes[0, 5]).to eq('%PDF-')
  end

  it 'saves the markdown PDF to a path the OS can re-read' do
    Dir.mktmpdir do |tmp|
      out = File.join(tmp, 'smoke.pdf')
      ok = PdfOxide::Creator.from_markdown('# Phase 2 smoke').save(out)
      expect(ok).to be(true)
      expect(File).to exist(out)
      expect(File.size(out)).to be > 512
      File.open(out, 'rb') { |f| expect(f.read(5)).to eq('%PDF-') }
    end
  end
end
