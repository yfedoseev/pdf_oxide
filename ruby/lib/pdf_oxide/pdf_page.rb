# frozen_string_literal: true

module PdfOxide
  # A page within a {PdfDocument}, identified by 0-based page index.
  #
  # Mirrors `fyi.oxide.pdf.PdfPage`.  Lightweight view — holds no
  # native handle of its own; it borrows from its parent document.
  # Operations after the parent's `#close` raise `InvalidStateError`.
  #
  # Construct via {PdfDocument#page} or {PdfDocument#pages}.
  class PdfPage
    # @return [PdfDocument] the owning document.
    attr_reader :parent

    # @return [Integer] 0-based page index.
    attr_reader :index

    # @api private (use {PdfDocument#page})
    def initialize(parent, index)
      raise ::PdfOxide::ArgumentError, 'parent cannot be nil' if parent.nil?

      @parent = parent
      @index  = index
    end

    # @return [Float] page width in PDF user-space units.
    def width
      media_box[:width]
    end

    # @return [Float] page height in PDF user-space units.
    def height
      media_box[:height]
    end

    # @return [Hash] { x:, y:, width:, height: } in PDF user-space.
    #   v0.3.55 limitation: pdf_oxide doesn't yet expose a public
    #   per-page media-box accessor through the C ABI; the canonical
    #   route is `pdf_render_page_fit`'s implicit dimensions.  Returns
    #   a zero-rect placeholder for now — mirrors PdfPage::cropBox()
    #   in Java which also currently defers crop-box access.
    def media_box
      { x: 0.0, y: 0.0, width: 0.0, height: 0.0 }
    end

    # @return [Hash] { x:, y:, width:, height: } — crop box, falling
    #   back to {#media_box} when /CropBox is absent (Java parity).
    def crop_box
      media_box
    end

    # @return [Integer] page rotation in degrees.  v0.3.55: the C ABI
    #   doesn't yet expose a per-page rotation accessor — returns 0.
    def rotation
      0
    end

    # Extract this page's text.  Equivalent to `parent.extract_text(index)`.
    # @return [String]
    def text
      @parent.extract_text(@index)
    end

    # @return [String] short inspection-style label (`#<PdfOxide::PdfPage index=N>`).
    #   Use {#text} to get the extracted page text.
    def to_s
      "#<PdfOxide::PdfPage index=#{@index}>"
    end
    alias inspect to_s
  end
end
