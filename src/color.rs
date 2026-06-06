//! Colour management for PDF rendering and image extraction.
//!
//! PDF (ISO 32000-1:2008) permits colour to be specified in a variety of
//! colour spaces — device-dependent (`DeviceGray`, `DeviceRGB`,
//! `DeviceCMYK`) and device-independent (`CalGray`, `CalRGB`, `Lab`,
//! `ICCBased`). Per §8.6.5.5 a conforming reader *shall* support the
//! ICC specification version required by the PDF version it claims to
//! accept (PDF 1.7 requires ICC.1:2004-10) and process embedded ICC
//! profiles rather than falling back to the `/Alternate` colour space
//! when the profile is understandable.
//!
//! The module is structured in three layers:
//!
//! 1. **Header parsing** — pure Rust, no dependencies. Extracts just
//!    enough from the 128-byte ICC header to decide whether we can
//!    handle a profile (version, device class, input colour space,
//!    profile connection space).
//! 2. **Rendering intent** — PDF-spec names → CMM-friendly enum. Used
//!    everywhere a colour conversion is performed (images, text, vector
//!    rendering). Default per §8.6.5.8 is `RelativeColorimetric`.
//! 3. **Transforms** — builds a source-profile → sRGB transform
//!    honouring a rendering intent. When the `icc` feature is enabled
//!    qcms compiles the embedded profile into a real colourimetric
//!    transform; otherwise transforms fall back to the §10.3.5
//!    additive-clamp formula so callers don't have to care whether a
//!    CMM is linked in.

#![forbid(unsafe_code)]

use std::sync::Arc;

/// PDF rendering intents, per ISO 32000-1:2008 §8.6.5.8 Table 70.
///
/// Specified on image XObjects (`/Intent`), in the graphics state
/// (`/RI` or via the `ri` operator), and implicitly wherever CIE-based
/// colour values must be reconciled with an output device's gamut.
///
/// Per §8.6.5.8: "If a conforming reader does not recognize the
/// specified name, it shall use the RelativeColorimetric intent by
/// default."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize)]
pub enum RenderingIntent {
    /// Preserve perceptual relationships; may modify in-gamut colours
    /// to maintain their relationship with out-of-gamut colours.
    Perceptual,
    /// Default per ISO 32000-1:2008 §8.6.5.8. Map source white to
    /// destination white; preserve in-gamut colours exactly, clip
    /// out-of-gamut.
    #[default]
    RelativeColorimetric,
    /// Preserve colour saturation over precise colourimetric values.
    Saturation,
    /// No white-point adaptation; preserve absolute colourimetric
    /// values across source and destination.
    AbsoluteColorimetric,
}

impl RenderingIntent {
    /// Resolve a PDF intent name to the enum, applying the spec's
    /// "unrecognised → RelativeColorimetric" fallback rule.
    pub fn from_pdf_name(name: &str) -> Self {
        match name {
            "Perceptual" => Self::Perceptual,
            "Saturation" => Self::Saturation,
            "AbsoluteColorimetric" => Self::AbsoluteColorimetric,
            // §8.6.5.8: unrecognized names fall through to RelativeColorimetric.
            _ => Self::RelativeColorimetric,
        }
    }
}

/// ICC profile header (first 128 bytes, per ICC.1:2004-10 §7.2).
///
/// We parse a minimal subset — enough to decide whether a profile is
/// usable and what colour space it expects on input/output. The rest
/// of the profile (tag table, curves, LUTs) is handed verbatim to the
/// CMM when one is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IccHeader {
    /// Profile format version, packed major.minor.bugfix from header bytes 8-11.
    pub version: u32,
    /// `deviceClass` signature (header bytes 12-15) —
    /// 'scnr', 'mntr', 'prtr', 'link', 'spac', 'abst', 'nmcl'.
    pub device_class: [u8; 4],
    /// `colorSpace` signature (header bytes 16-19) —
    /// 'GRAY', 'RGB ', 'CMYK', 'Lab ', 'XYZ ', …
    pub color_space: [u8; 4],
    /// Profile connection space (header bytes 20-23) — typically
    /// 'XYZ ' or 'Lab '.
    pub pcs: [u8; 4],
}

impl IccHeader {
    /// The ICC signature at bytes 36-39 must be 'acsp' for a valid profile.
    const ACSP: [u8; 4] = *b"acsp";

    /// Parse the 128-byte ICC header. Returns `None` if the input is
    /// too short or the `acsp` signature is missing.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 128 {
            return None;
        }
        // Validate the ICC signature — without this almost any random
        // byte sequence would be accepted as a "profile".
        let sig = [bytes[36], bytes[37], bytes[38], bytes[39]];
        if sig != Self::ACSP {
            return None;
        }
        let version = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let device_class = [bytes[12], bytes[13], bytes[14], bytes[15]];
        let color_space = [bytes[16], bytes[17], bytes[18], bytes[19]];
        let pcs = [bytes[20], bytes[21], bytes[22], bytes[23]];
        Some(Self {
            version,
            device_class,
            color_space,
            pcs,
        })
    }

    /// Number of components implied by the input colour space
    /// signature. Returns `None` for signatures we don't recognise —
    /// callers should then cross-check against the `/N` entry the PDF
    /// dictionary advertised and reject the profile if they disagree.
    pub fn input_components(&self) -> Option<u8> {
        match &self.color_space {
            b"GRAY" => Some(1),
            b"RGB " => Some(3),
            b"Lab " | b"XYZ " => Some(3),
            b"CMYK" => Some(4),
            _ => None,
        }
    }
}

/// An embedded ICC profile, ready to be handed to a colour management
/// module. The raw bytes are retained so the CMM can build its own
/// compiled transform from them; `header` is the eagerly-parsed
/// 128-byte prefix for cheap interrogation without re-parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IccProfile {
    /// Full profile bytes (post-FlateDecode). May be many hundreds of
    /// KiB for real CMYK production profiles.
    bytes: Arc<Vec<u8>>,
    /// Number of input components from the colour-space dictionary's
    /// `/N` entry. The spec mandates this match the profile header's
    /// colour-space signature; we treat the dict as authoritative when
    /// they disagree so malformed profiles can't resize downstream
    /// buffers unexpectedly.
    n_components: u8,
    header: IccHeader,
}

impl IccProfile {
    /// Parse profile bytes, cross-checking the dictionary's declared
    /// component count against the header's colour-space signature.
    /// Returns `None` if the header is invalid or the component counts
    /// contradict each other.
    pub fn parse(bytes: Vec<u8>, declared_n: u8) -> Option<Self> {
        let header = IccHeader::parse(&bytes)?;
        // Cross-check: the header's colorSpace signature must imply the
        // same component count the PDF dict said. PDF 32000-1 §8.6.5.5:
        // "N shall match the number of components actually in the ICC
        // profile." Reject mismatches instead of guessing.
        if let Some(hdr_n) = header.input_components() {
            if hdr_n != declared_n {
                return None;
            }
        }
        Some(Self {
            bytes: Arc::new(bytes),
            n_components: declared_n,
            header,
        })
    }

    /// Raw profile bytes, post-decompression. The CMM layer consumes
    /// these directly when building a compiled transform.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Input component count (1, 3, or 4) as declared by the PDF
    /// dictionary and cross-checked against the profile header.
    pub fn n_components(&self) -> u8 {
        self.n_components
    }

    /// Parsed 128-byte ICC header — cheap to access, no re-parsing cost.
    pub fn header(&self) -> &IccHeader {
        &self.header
    }

    /// Hash the profile bytes for use as a transform-cache key. Two
    /// profiles with identical bytes produce identical compiled
    /// transforms, so this is sufficient.
    pub fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.bytes.hash(&mut h);
        h.finish()
    }
}

/// A compiled source-profile → sRGB transform for a given intent.
///
/// With the `icc` feature enabled the inner representation is a real
/// qcms transform; without it, the transform falls back to ISO 32000-1
/// §10.3.5's additive-clamp formula so the API stays the same whether
/// or not a CMM is linked in. This lets downstream callers invoke the
/// same `convert_*` methods regardless of build configuration.
pub struct Transform {
    /// The profile we compiled from (kept for diagnostics / re-use).
    source_profile: Arc<IccProfile>,
    intent: RenderingIntent,
    #[cfg(feature = "icc")]
    inner: Option<QcmsHolder>,
}

#[cfg(feature = "icc")]
struct QcmsHolder {
    /// Source → sRGB8 compiled transform. The source component type is
    /// whatever the ICC profile advertised (CMYK/RGB/Gray); we build at
    /// most one transform per `Transform` instance since PDF images
    /// carry a single source profile and are decoded in one colour
    /// space at a time.
    inner: qcms::Transform,
}

#[cfg(feature = "icc")]
fn qcms_intent(intent: RenderingIntent) -> qcms::Intent {
    match intent {
        RenderingIntent::Perceptual => qcms::Intent::Perceptual,
        RenderingIntent::RelativeColorimetric => qcms::Intent::RelativeColorimetric,
        RenderingIntent::Saturation => qcms::Intent::Saturation,
        RenderingIntent::AbsoluteColorimetric => qcms::Intent::AbsoluteColorimetric,
    }
}

#[cfg(feature = "icc")]
fn try_build_qcms_holder(
    profile_bytes: &[u8],
    n_components: u8,
    intent: RenderingIntent,
) -> Option<QcmsHolder> {
    let src = qcms::Profile::new_from_slice(profile_bytes, false)?;
    let dst = qcms::Profile::new_sRGB();
    let i = qcms_intent(intent);

    // Build the source → sRGB transform matching the profile's declared
    // input component type. Unrecognised counts fall through to `None`
    // so the caller uses the §10.3.5 fallback.
    let src_ty = match n_components {
        1 => qcms::DataType::Gray8,
        3 => qcms::DataType::RGB8,
        4 => qcms::DataType::CMYK,
        _ => return None,
    };
    qcms::Transform::new_to(&src, &dst, src_ty, qcms::DataType::RGB8, i)
        .map(|inner| QcmsHolder { inner })
}

impl Transform {
    /// Build a source→sRGB transform for the given profile and intent.
    /// When the `icc` feature is on, qcms compiles the embedded profile
    /// into a real colourimetric transform; otherwise the transform is
    /// a thin wrapper around the §10.3.5 additive-clamp fallback.
    ///
    /// Per-page caching of the compiled transform lives on
    /// `crate::rendering::resolution::IccTransformCache`; this method
    /// is the underlying builder the cache calls into on a miss.
    pub fn new_srgb_target(profile: Arc<IccProfile>, intent: RenderingIntent) -> Self {
        #[cfg(feature = "icc")]
        {
            let inner = try_build_qcms_holder(profile.bytes(), profile.n_components(), intent);
            Self {
                source_profile: profile,
                intent,
                inner,
            }
        }
        #[cfg(not(feature = "icc"))]
        {
            Self {
                source_profile: profile,
                intent,
            }
        }
    }

    /// Convert one CMYK sample to RGB. With a qcms transform available
    /// this runs the CMM; otherwise it falls back to §10.3.5.
    pub fn convert_cmyk_pixel(&self, c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
        #[cfg(feature = "icc")]
        {
            if let Some(holder) = &self.inner {
                if self.source_profile.n_components() == 4 {
                    let src = [c, m, y, k];
                    let mut dst = [0u8; 3];
                    holder.inner.convert(&src, &mut dst);
                    return dst;
                }
            }
        }
        crate::extractors::images::cmyk_pixel_to_rgb(c, m, y, k)
    }

    /// Convert a packed CMYK byte slice to RGB. When qcms is available
    /// this is a single bulk `qcms::Transform::convert` call; otherwise
    /// it falls back to the per-pixel §10.3.5 formula.
    pub fn convert_cmyk_buffer(&self, cmyk: &[u8]) -> Vec<u8> {
        #[cfg(feature = "icc")]
        {
            if let Some(holder) = &self.inner {
                if self.source_profile.n_components() == 4 {
                    let pixels = cmyk.len() / 4;
                    let mut out = vec![0u8; pixels * 3];
                    holder.inner.convert(cmyk, &mut out);
                    return out;
                }
            }
        }
        let mut out = Vec::with_capacity((cmyk.len() / 4) * 3);
        for ch in cmyk.chunks_exact(4) {
            let rgb = self.convert_cmyk_pixel(ch[0], ch[1], ch[2], ch[3]);
            out.extend_from_slice(&rgb);
        }
        out
    }

    /// Convert a packed RGB byte slice through the source profile to
    /// sRGB. Useful for `/ICCBased` N=3 colour spaces (Adobe RGB,
    /// ProPhoto, wide-gamut cameras …). When qcms is unavailable or
    /// the profile isn't RGB, returns the input unchanged (the input
    /// is already assumed to be sRGB-like).
    pub fn convert_rgb_buffer(&self, rgb: &[u8]) -> Vec<u8> {
        #[cfg(feature = "icc")]
        {
            if let Some(holder) = &self.inner {
                if self.source_profile.n_components() == 3 {
                    let mut out = vec![0u8; rgb.len()];
                    holder.inner.convert(rgb, &mut out);
                    return out;
                }
            }
        }
        rgb.to_vec()
    }

    /// Convert a packed grayscale byte slice through the source profile
    /// to sRGB (outputs 3 bytes per input byte). When qcms is
    /// unavailable or the profile isn't Gray, replicates the grayscale
    /// channel into RGB.
    pub fn convert_gray_buffer(&self, gray: &[u8]) -> Vec<u8> {
        #[cfg(feature = "icc")]
        {
            if let Some(holder) = &self.inner {
                if self.source_profile.n_components() == 1 {
                    let mut out = vec![0u8; gray.len() * 3];
                    holder.inner.convert(gray, &mut out);
                    return out;
                }
            }
        }
        let mut out = Vec::with_capacity(gray.len() * 3);
        for &g in gray {
            out.extend_from_slice(&[g, g, g]);
        }
        out
    }

    /// Component count the source profile accepts (1, 3, or 4). Callers
    /// use this to pick the matching `convert_*_buffer` method for a
    /// given pixel format and to suppress mismatched transforms.
    pub fn source_n_components(&self) -> u8 {
        self.source_profile.n_components()
    }

    /// Whether a real ICC transform is in play (vs the §10.3.5 fallback).
    pub fn has_cmm(&self) -> bool {
        #[cfg(feature = "icc")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "icc"))]
        {
            false
        }
    }
}

impl std::fmt::Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transform")
            .field("intent", &self.intent)
            .field("profile_bytes", &self.source_profile.bytes.len())
            .field("n_components", &self.source_profile.n_components)
            .field("cmm_live", &self.has_cmm())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid ICC header — just enough to satisfy `parse`.
    /// Bytes 0-3: size; 4-7: CMM; 8-11: version (4.2.0.0); 12-15: devClass;
    /// 16-19: colour space; 20-23: PCS; … 36-39: 'acsp'. Remaining bytes
    /// unused for this test.
    fn minimal_header(cs: &[u8; 4], n_bytes: usize) -> Vec<u8> {
        let mut v = vec![0u8; n_bytes.max(128)];
        v[8..12].copy_from_slice(&0x04200000u32.to_be_bytes());
        v[12..16].copy_from_slice(b"prtr");
        v[16..20].copy_from_slice(cs);
        v[20..24].copy_from_slice(b"Lab ");
        v[36..40].copy_from_slice(b"acsp");
        v
    }

    #[test]
    fn header_parse_requires_acsp_signature() {
        let mut bytes = minimal_header(b"CMYK", 128);
        bytes[36..40].copy_from_slice(b"xxxx");
        assert!(IccHeader::parse(&bytes).is_none());
    }

    #[test]
    fn header_parse_rejects_short_input() {
        let bytes = vec![0u8; 127];
        assert!(IccHeader::parse(&bytes).is_none());
    }

    #[test]
    fn header_identifies_cmyk_as_4_components() {
        let bytes = minimal_header(b"CMYK", 128);
        let h = IccHeader::parse(&bytes).expect("valid header");
        assert_eq!(h.input_components(), Some(4));
        assert_eq!(&h.color_space, b"CMYK");
        assert_eq!(&h.device_class, b"prtr");
    }

    #[test]
    fn profile_parse_rejects_n_mismatch() {
        // Header advertises CMYK (4 components) but dictionary declares N=3.
        // PDF §8.6.5.5 requires these to agree.
        let bytes = minimal_header(b"CMYK", 128);
        assert!(IccProfile::parse(bytes, 3).is_none());
    }

    #[test]
    fn profile_parse_accepts_matching_n() {
        let bytes = minimal_header(b"CMYK", 128);
        let p = IccProfile::parse(bytes, 4).expect("should parse");
        assert_eq!(p.n_components(), 4);
    }

    #[test]
    fn intent_default_is_relative_colorimetric() {
        assert_eq!(RenderingIntent::default(), RenderingIntent::RelativeColorimetric);
    }

    #[test]
    fn intent_from_pdf_name_falls_back_to_relative_colorimetric() {
        // §8.6.5.8: unrecognized names fall through.
        assert_eq!(
            RenderingIntent::from_pdf_name("WhateverNotReal"),
            RenderingIntent::RelativeColorimetric,
        );
        assert_eq!(RenderingIntent::from_pdf_name("Perceptual"), RenderingIntent::Perceptual,);
        assert_eq!(RenderingIntent::from_pdf_name("Saturation"), RenderingIntent::Saturation,);
        assert_eq!(
            RenderingIntent::from_pdf_name("AbsoluteColorimetric"),
            RenderingIntent::AbsoluteColorimetric,
        );
    }

    #[test]
    fn phase1_transform_preserves_srgb_white() {
        let bytes = minimal_header(b"CMYK", 128);
        let p = Arc::new(IccProfile::parse(bytes, 4).unwrap());
        let t = Transform::new_srgb_target(p, RenderingIntent::RelativeColorimetric);
        // CMYK(0,0,0,0) → sRGB white under any sensible transform.
        assert_eq!(t.convert_cmyk_pixel(0, 0, 0, 0), [255, 255, 255]);
        // CMYK(255,255,255,255) → sRGB black under the §10.3.5 fallback.
        assert_eq!(t.convert_cmyk_pixel(255, 255, 255, 255), [0, 0, 0]);
    }
}
