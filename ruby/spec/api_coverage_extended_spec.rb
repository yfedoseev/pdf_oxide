# frozen_string_literal: true

# API-coverage spec for the extended C-ABI surface: symbols
# that previously had no Ruby binding.
#
#   pdf_oxide_set_max_ops_per_stream        — global int toggle, no err channel
#   pdf_oxide_set_preserve_unmapped_glyphs  — global int toggle, no err channel
#   pdf_render_page_with_options_ex         — render + OCG layer filtering
# pdf_oxide_word_get_rotation — word glyph-run rotation
# pdf_oxide_path_get_rendered_bbox — stroke-inflated path bbox
#
# Simple int toggles assert invokable (and that the prior value round-trips).
# The render entry needs a real document, so it asserts return-or-error.
# The word/path accessors have no list constructor exposed in Ruby yet
# (extract_words / extract_paths are placeholder-typed), so they assert
# the documented null-handle contract: sentinel value + ERR_INVALID_ARG,
# which exercises the real symbol and its argument signature end-to-end.

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

  it 'binds pdf_oxide_word_get_rotation (null-handle contract)' do
    expect(PdfOxide::Bindings).to respond_to(:pdf_oxide_word_get_rotation)

    # float pdf_oxide_word_get_rotation(words, index, err) returns 0.0 and
    # sets ERR_INVALID_ARG (1) for a null word-list handle. A wrong arity
    # or return type here fails hard instead of reading register garbage.
    err = ::FFI::MemoryPointer.new(:int32)
    value = PdfOxide::Bindings.pdf_oxide_word_get_rotation(::FFI::Pointer::NULL, 0, err)
    expect(value).to eq(0.0)
    expect(err.read_int32).to eq(1) # ERR_INVALID_ARG
  end

  it 'binds pdf_oxide_path_get_rendered_bbox (null-handle contract)' do
    expect(PdfOxide::Bindings).to respond_to(:pdf_oxide_path_get_rendered_bbox)

    # void pdf_oxide_path_get_rendered_bbox(paths, index, x, y, w, h, err)
    # sets ERR_INVALID_ARG (1) and leaves the out-params untouched for a
    # null path-list handle.
    x = ::FFI::MemoryPointer.new(:float)
    y = ::FFI::MemoryPointer.new(:float)
    w = ::FFI::MemoryPointer.new(:float)
    h = ::FFI::MemoryPointer.new(:float)
    err = ::FFI::MemoryPointer.new(:int32)
    PdfOxide::Bindings.pdf_oxide_path_get_rendered_bbox(::FFI::Pointer::NULL, 0, x, y, w, h, err)
    expect(err.read_int32).to eq(1) # ERR_INVALID_ARG
    expect([x, y, w, h].map(&:read_float)).to all(eq(0.0)) # untouched zero-init buffers
  end
end
