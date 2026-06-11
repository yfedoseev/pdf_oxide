# Vertical-CJK rendering fixtures

These fixtures exercise the page-render path for PDFs that reference Adobe
predefined CIDFonts (`Ryumin-Light`, `GothicBBB-Medium`, `HeiseiKakuGo-W5`)
without embedding the glyph outlines. ISO 32000-2:2020 §9.7.5.2 mandates
conforming reader support for the Adobe-Japan1 / Adobe-GB1 / Adobe-CNS1 /
Adobe-Korea1 character collections; before this work the renderer dropped
every glyph to `.notdef` and produced a blank page for these documents.

## Files

### `jo.pdf`

Single-page DVIPDFMx-emitted document containing a short Japanese poem in
vertical (`Identity-V`) writing mode. Uses `Ryumin-Light` (Adobe-Japan1) with
no embedded outlines, no ToUnicode CMap. The classic minimal repro for the
"vertical CJK PDF renders blank" failure mode.

- Producer: `dvipdfmx (20031116)`
- Creator: `TeX output 2004.10.26:1808`
- Source: pdfminer.six test corpus
  (<https://github.com/pdfminer/pdfminer.six/blob/master/samples/jo.pdf>)
- License: MIT — redistributable under the same terms as pdfminer.six itself.

### `kampo.pdf`

Multi-page Japanese pharmaceutical reference text. Uses `Ryumin-Light` +
`GothicBBB-Medium` (both Adobe-Japan1) in both horizontal and vertical
writing modes, plus `HeiseiKakuGo-W5` references; a single embedded `Type1C`
subset (`GMALPM+DFHSMincho-W3G014`) provides Latin glyphs but the CJK faces
all rely on reader-supplied glyph outlines.

- Producer: `Acrobat Distiller 5.0.5 for Macintosh`
- Source: pdfminer.six test corpus
  (<https://github.com/pdfminer/pdfminer.six/blob/master/samples/kampo.pdf>)
- License: MIT — redistributable under the same terms as pdfminer.six itself.

## Glyph supply

When the `cjk-render-fallback` cargo feature is enabled, glyphs for these
predefined CIDFonts are supplied from the bundled Droid Sans Fallback font
(`src/fonts/assets/DroidSansFallbackFull.ttf`, Apache-2.0). Droid Sans
Fallback is sans-serif; Mincho / Gothic distinction is not preserved.
Vertical-form glyph variants (`vert` / `vrt2` features) are not provided —
the renderer uses the same glyph shape for horizontal and vertical
substitution and relies on the document's own writing-mode advance to lay
the glyphs out vertically.
