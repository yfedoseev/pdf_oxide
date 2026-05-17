//! Destructive-redaction orchestration for a single content stream
//! (#231, T11 — the parse → prune → re-serialize → overlay pipeline that
//! turns the pure primitives into a working redaction of real bytes).
//!
//! Pipeline (ISO 32000-1:2008 §7.8.2 content-stream model):
//!
//! 1. `parse_content_stream` → operators.
//! 2. `text_engine::redact_text_stream` removes every glyph touching a
//!    region and re-emits survivors with absolute `Tm`, no `TJ` deltas
//!    (G1/G2). A composite/Type0/unknown font ⇒ **hard refusal**
//!    (`Err`), never a silent pass-through (feature plan §9 risk 6).
//! 3. `serialize` re-serializes survivors (binary-safe strings, G6).
//! 4. `overlay` appends one opaque block per region *after* the pruned
//!    content so the redacted area is visibly marked (G7) and the
//!    underlying bytes are already gone — not merely covered.
//!
//! This module owns no document I/O: it is `bytes + regions → bytes`,
//! so the security guarantee (secret absent from the *output bytes*) is
//! directly fixture-testable here without the writer/save path. Wiring a
//! page's real content + fonts and forcing full-rewrite GC is the
//! `DocumentEditor` step (T12); residual-object destruction (G6) is
//! enforced there by the existing garbage-collected full rewrite.

use super::options::{RedactionOptions, RedactionReport};
use super::overlay::region_overlay_ops;
use super::region::RegionSet;
use super::serialize::serialize_operator;
use super::text_engine::{redact_text_stream, FontMetrics};
use crate::content::parser::parse_content_stream;
use crate::error::{Error, Result};
use crate::fonts::FontInfo;
use std::collections::HashMap;
use std::sync::Arc;

/// [`FontMetrics`] over the real per-page `FontInfo` map (resource name
/// → font). The single source of glyph widths is `FontInfo`
/// (`get_glyph_width`) — the *same* metric the text extractor uses, so
/// the redactor's geometry cannot diverge from extraction (DRY; feature
/// plan §9 risk 1). An unknown font, or any composite/Type0 font, is
/// reported non-simple so the engine refuses rather than under-redacts.
pub struct FontInfoMetrics {
    fonts: HashMap<String, Arc<FontInfo>>,
}

impl FontInfoMetrics {
    /// Build from a page's resolved `/Font` resources (resource name →
    /// parsed [`FontInfo`]).
    pub fn new(fonts: HashMap<String, Arc<FontInfo>>) -> Self {
        Self { fonts }
    }
}

impl FontMetrics for FontInfoMetrics {
    fn width(&self, font: &str, code: u32) -> f32 {
        match self.fonts.get(font) {
            // `get_glyph_width` returns /1000 glyph-space units.
            Some(fi) => fi.get_glyph_width(code.min(u32::from(u16::MAX)) as u16),
            // Unknown font: full-em over-estimate (the show is refused
            // anyway via `is_simple`, but never under-size a box).
            None => 1000.0,
        }
    }

    fn is_simple(&self, font: &str) -> bool {
        // Only single-byte simple fonts have a reliable byte→code→width
        // path here. Type0/CID and unknown ⇒ refuse (fail closed).
        match self.fonts.get(font) {
            Some(fi) => fi.subtype != "Type0",
            None => false,
        }
    }
}

/// Destructively redact the text of one content stream.
///
/// Returns the rewritten stream bytes (pruned content + opaque overlays)
/// and a [`RedactionReport`]. The returned bytes are guaranteed to
/// contain no glyph whose box intersected an (edge-padded) region — the
/// secret is *removed*, then an opaque block is drawn over the area.
///
/// # Errors
/// - [`Error::Unsupported`] — a text show used a composite/Type0/unknown
///   font while regions exist; redaction is **refused** rather than risk
///   a silent under-redaction (feature plan §9 risk 6, fail closed).
/// - [`Error::ParseError`] — the content stream did not parse.
pub fn redact_content_stream(
    content: &[u8],
    regions: &RegionSet,
    opts: &RedactionOptions,
    fonts: &dyn FontMetrics,
) -> Result<(Vec<u8>, RedactionReport)> {
    let ops = parse_content_stream(content)?;
    let te = redact_text_stream(&ops, regions, opts.edge_padding, fonts);

    if te.unsupported_font {
        return Err(Error::Unsupported(
            "destructive text redaction of composite/Type0 font content is not yet \
             supported; refusing rather than risk leaving recoverable text"
                .to_string(),
        ));
    }

    let mut body = Vec::with_capacity(content.len());
    for op in &te.operators {
        serialize_operator(&mut body, op);
    }

    for region in &regions.regions {
        body.extend_from_slice(&region_overlay_ops(region, opts));
    }

    let report = RedactionReport {
        regions: regions.len(),
        glyphs_removed: te.glyphs_removed,
        annotations_removed: 0,
        fonts_scrubbed: 0,
        bytes_removed: te.bytes_removed,
        ..RedactionReport::default()
    };
    Ok((body, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redaction::region::RedactionRegion;

    /// 10-unit-wide simple font (so a `BT 1 0 0 1 X Y Tm` places glyph i
    /// at page x = X + 10·i for size 10... here width 500/1000·10 = 5pt).
    struct Stub;
    impl FontMetrics for Stub {
        fn width(&self, _f: &str, _c: u32) -> f32 {
            500.0
        }
    }

    /// §7.1 oracle (simplest form): the secret bytes must not appear
    /// anywhere in the (uncompressed) output content stream.
    fn assert_absent(out: &[u8], secret: &[u8]) {
        assert!(
            out.windows(secret.len()).all(|w| w != secret),
            "secret {:?} still present in redacted output: {:?}",
            String::from_utf8_lossy(secret),
            String::from_utf8_lossy(out)
        );
    }

    fn one_region(x0: f32, y0: f32, x1: f32, y1: f32) -> RegionSet {
        let mut rs = RegionSet::new(0);
        rs.push(RedactionRegion::from_rect(x0, y0, x1, y1, Some([0.0, 0.0, 0.0])));
        rs
    }

    const SECRET_DOC: &[u8] = b"BT\n/F1 10 Tf\n1 0 0 1 100 700 Tm\n(TOPSECRET) Tj\nET\n";

    #[test]
    fn secret_fully_in_region_is_removed_and_overlaid() {
        // "TOPSECRET" 9 glyphs ×5pt from x=100 → 100..145 at y≈700.
        let regions = one_region(90.0, 695.0, 160.0, 715.0);
        let (out, report) =
            redact_content_stream(SECRET_DOC, &regions, &RedactionOptions::default(), &Stub)
                .unwrap();
        assert_absent(&out, b"TOPSECRET");
        assert_absent(&out, b"SECRET");
        assert_eq!(report.glyphs_removed, 9);
        assert_eq!(report.regions, 1);
        assert!(report.bytes_removed > 0);
        // An opaque overlay block is present (G7): the overlay emits
        // `q\n<r> <g> <b> rg\n<x> <y> <w> <h> re\nf\nQ\n`.
        let s = String::from_utf8_lossy(&out);
        assert!(
            s.contains("rg\n") && s.contains(" re\n") && s.contains("\nf\n"),
            "overlay missing in: {s}"
        );
    }

    #[test]
    fn public_text_outside_region_survives_verbatim() {
        let doc = b"BT\n/F1 10 Tf\n1 0 0 1 100 700 Tm\n(PUBLIC) Tj\nET\n";
        let regions = one_region(0.0, 0.0, 5.0, 5.0); // far away
        let (out, report) =
            redact_content_stream(doc, &regions, &RedactionOptions::default(), &Stub).unwrap();
        assert_eq!(report.glyphs_removed, 0);
        assert!(
            out.windows(6).any(|w| w == b"PUBLIC"),
            "public text must survive: {}",
            String::from_utf8_lossy(&out)
        );
    }

    #[test]
    fn straddling_secret_partially_removed_public_kept() {
        // "PUBSECRET": region from x≈124.5 onward removes "SECRET",
        // keeps "PUB". Glyph i at x = 100 + 5i: P100 U105 B110 S115…
        let doc = b"BT\n/F1 10 Tf\n1 0 0 1 100 700 Tm\n(PUBSECRET) Tj\nET\n";
        let regions = one_region(120.0, 695.0, 400.0, 715.0);
        let (out, report) =
            redact_content_stream(doc, &regions, &RedactionOptions::default(), &Stub).unwrap();
        assert_absent(&out, b"SECRET");
        assert!(report.glyphs_removed >= 6);
        assert!(
            out.windows(3).any(|w| w == b"PUB"),
            "public prefix must survive: {}",
            String::from_utf8_lossy(&out)
        );
    }

    #[test]
    fn composite_font_is_refused_not_under_redacted() {
        struct Composite;
        impl FontMetrics for Composite {
            fn width(&self, _f: &str, _c: u32) -> f32 {
                500.0
            }
            fn is_simple(&self, _f: &str) -> bool {
                false
            }
        }
        let regions = one_region(0.0, 0.0, 1000.0, 1000.0);
        let err =
            redact_content_stream(SECRET_DOC, &regions, &RedactionOptions::default(), &Composite)
                .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "expected refusal, got {err:?}");
    }

    #[test]
    fn no_regions_keeps_content_and_draws_nothing() {
        let (out, report) = redact_content_stream(
            SECRET_DOC,
            &RegionSet::new(0),
            &RedactionOptions::default(),
            &Stub,
        )
        .unwrap();
        assert_eq!(report.glyphs_removed, 0);
        assert_eq!(report.regions, 0);
        // Nothing redacted, no overlay; the text is intact.
        assert!(out.windows(9).any(|w| w == b"TOPSECRET"));
    }

    #[test]
    fn malformed_stream_is_a_clean_error_not_a_panic() {
        // parse_content_stream is tolerant; ensure no panic and that if
        // it does parse, an empty/garbage stream yields no secret.
        let regions = one_region(0.0, 0.0, 1000.0, 1000.0);
        let _ = redact_content_stream(
            b"q Q (unbalanced",
            &regions,
            &RedactionOptions::default(),
            &Stub,
        );
        let _ = redact_content_stream(b"", &regions, &RedactionOptions::default(), &Stub);
    }
}
