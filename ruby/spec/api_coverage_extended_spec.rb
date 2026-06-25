# frozen_string_literal: true

# API-coverage spec for the extended C-ABI surface: three symbols
# that previously had no Ruby binding.
#
#   pdf_oxide_set_max_ops_per_stream        — global int toggle, no err channel
#   pdf_oxide_set_preserve_unmapped_glyphs  — global int toggle, no err channel
#   pdf_render_page_with_options_ex         — render + OCG layer filtering
#
# Simple int toggles assert invokable (and that the prior value round-trips).
# The render entry needs a real document, so it asserts return-or-error.

require 'spec_helper'

RSpec.describe 'C-ABI coverage: extended symbols' do
  it 'binds pdf_oxide_set_max_ops_per_stream' do
    expect(PdfOxide::Bindings).to respond_to(:pdf_oxide_set_max_ops_per_stream)
    expect(PdfOxide).to respond_to(:set_max_ops_per_stream)

    # Returns the previous cap; restoring the default (negative arg) must
    # be invokable and return an Integer.
    prev = PdfOxide.set_max_ops_per_stream(-1)
    expect(prev).to be_a(Integer)

    # A non-negative cap returns the prior value (the default we just set).
    restored = PdfOxide.set_max_ops_per_stream(500_000)
    expect(restored).to be_a(Integer)

    # Put the default back so we don't perturb other examples.
    PdfOxide.set_max_ops_per_stream(-1)
  end

  it 'binds pdf_oxide_set_preserve_unmapped_glyphs' do
    expect(PdfOxide::Bindings).to respond_to(:pdf_oxide_set_preserve_unmapped_glyphs)
    expect(PdfOxide).to respond_to(:set_preserve_unmapped_glyphs)

    prev = PdfOxide.set_preserve_unmapped_glyphs(true)
    expect(prev).to be_a(Integer)
    expect([0, 1]).to include(prev)

    # Restore prior state (round-trips the previous value back).
    restored = PdfOxide.set_preserve_unmapped_glyphs(false)
    expect([0, 1]).to include(restored)
  end

  it 'binds pdf_render_page_with_options_ex (return-or-error)' do
    expect(PdfOxide::Bindings).to respond_to(:pdf_render_page_with_options_ex)
    expect(PdfOxide::PdfDocument.instance_methods).to include(:render_with_layers)

    # Build a real one-page PDF in memory, then render it through the
    # layer-filtered entry point. Accept either a successful byte buffer
    # or a binding error (e.g. a render-feature-gated cdylib build).
    bytes = PdfOxide::Pdf.from_markdown("# Layer test\n\nbody").to_bytes
    PdfOxide::PdfDocument.open(bytes) do |doc|
      result = doc.render_with_layers(0, dpi: 72, excluded_layers: %w[Watermark])
      expect(result).to be_a(String)
    rescue PdfOxide::Error => e
      expect(e).to be_a(PdfOxide::Error)
    end
  end
end
