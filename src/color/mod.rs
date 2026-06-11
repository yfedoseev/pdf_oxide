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
//! The module is structured in four layers:
//!
//! 1. **Header parsing** — pure Rust, no dependencies. Extracts just
//!    enough from the 128-byte ICC header to decide whether we can
//!    handle a profile (version, device class, input colour space,
//!    profile connection space).
//! 2. **Rendering intent** — PDF-spec names → CMM-friendly enum. Used
//!    everywhere a colour conversion is performed (images, text, vector
//!    rendering). Default per §8.6.5.8 is `RelativeColorimetric`.
//! 3. **Backend abstraction** — see [`backend`]. Two CMMs ship behind
//!    feature flags: `icc-qcms` (pure-Rust default) and `icc-lcms2`
//!    (press-grade, opt-in C dep). Call sites in this module dispatch
//!    through [`backend::ActiveIccBackend`] so the rest of the codebase
//!    never imports qcms or lcms2 directly.
//! 4. **Transforms** — [`Transform`] (source-profile → sRGB) and
//!    [`CmykRetargetTransform`] (CMYK → CMYK retargeting via the
//!    destination profile's BToA, when the active backend supports
//!    it). The `convert_*` methods on `Transform` fall back to the
//!    §10.3.5 additive-clamp formula when no CMM is linked in, so
//!    downstream callers invoke the same surface regardless of build
//!    configuration.

#![forbid(unsafe_code)]

pub mod backend;

use std::sync::Arc;

#[allow(unused_imports)]
use backend::{ActiveIccBackend, IccBackend, TransformFlags};

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
/// Inner representation is whatever [`backend::ActiveIccBackend`]
/// resolves to at compile time: real qcms or lcms2 work when the
/// matching feature is enabled, otherwise the transform is a thin
/// wrapper around the ISO 32000-1 §10.3.5 additive-clamp formula so
/// the API stays the same whether or not a CMM is linked in.
pub struct Transform {
    /// The profile we compiled from (kept for diagnostics / re-use).
    source_profile: Arc<IccProfile>,
    intent: RenderingIntent,
    /// Cached source-component count, so the no-CMM fallback path
    /// doesn't dereference `source_profile.n_components()` on every
    /// per-pixel call.
    source_components: u8,
    inner: Option<<ActiveIccBackend as IccBackend>::SrgbTransform>,
}

impl Transform {
    /// Build a source→sRGB transform for the given profile and intent.
    /// When a backend is linked in (qcms or lcms2), the embedded
    /// profile is compiled into a real colourimetric transform;
    /// otherwise the transform is a thin wrapper around the §10.3.5
    /// additive-clamp fallback.
    ///
    /// Per-page caching of the compiled transform lives on
    /// `crate::rendering::resolution::IccTransformCache`; this method
    /// is the underlying builder the cache calls into on a miss.
    pub fn new_srgb_target(profile: Arc<IccProfile>, intent: RenderingIntent) -> Self {
        let n = profile.n_components();
        let inner = <ActiveIccBackend as IccBackend>::build_srgb_transform(
            &profile,
            intent,
            TransformFlags::press_default(),
        );
        Self {
            source_profile: profile,
            intent,
            source_components: n,
            inner,
        }
    }

    /// Convert one CMYK sample to RGB. With a real CMM transform
    /// available this runs the CMM; otherwise it falls back to §10.3.5.
    pub fn convert_cmyk_pixel(&self, c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
        if let Some(holder) = &self.inner {
            if self.source_components == 4 {
                if let Some(rgb) =
                    <ActiveIccBackend as IccBackend>::convert_cmyk_pixel(holder, [c, m, y, k])
                {
                    return rgb;
                }
            }
        }
        crate::extractors::images::cmyk_pixel_to_rgb(c, m, y, k)
    }

    /// Convert a packed CMYK byte slice to RGB. When the CMM is
    /// available this is a single batched call; otherwise it falls
    /// back to the per-pixel §10.3.5 formula.
    pub fn convert_cmyk_buffer(&self, cmyk: &[u8]) -> Vec<u8> {
        if let Some(holder) = &self.inner {
            if self.source_components == 4 {
                if let Some(out) =
                    <ActiveIccBackend as IccBackend>::convert_cmyk_buffer(holder, cmyk)
                {
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
    /// ProPhoto, wide-gamut cameras …). When no CMM is available or
    /// the profile isn't RGB, returns the input unchanged (the input
    /// is already assumed to be sRGB-like).
    pub fn convert_rgb_buffer(&self, rgb: &[u8]) -> Vec<u8> {
        if let Some(holder) = &self.inner {
            if self.source_components == 3 {
                if let Some(out) = <ActiveIccBackend as IccBackend>::convert_rgb_buffer(holder, rgb)
                {
                    return out;
                }
            }
        }
        rgb.to_vec()
    }

    /// Convert a packed grayscale byte slice through the source profile
    /// to sRGB (outputs 3 bytes per input byte). When no CMM is
    /// available or the profile isn't Gray, replicates the grayscale
    /// channel into RGB.
    pub fn convert_gray_buffer(&self, gray: &[u8]) -> Vec<u8> {
        if let Some(holder) = &self.inner {
            if self.source_components == 1 {
                if let Some(out) =
                    <ActiveIccBackend as IccBackend>::convert_gray_buffer(holder, gray)
                {
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
        self.source_components
    }

    /// Whether a real ICC transform is in play (vs the §10.3.5 fallback).
    pub fn has_cmm(&self) -> bool {
        self.inner.is_some()
    }
}

impl std::fmt::Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transform")
            .field("intent", &self.intent)
            .field("profile_bytes", &self.source_profile.bytes.len())
            .field("n_components", &self.source_components)
            .field("cmm_live", &self.has_cmm())
            .field("backend", &backend::active_backend_name())
            .finish()
    }
}

/// A compiled CMYK → CMYK retargeting transform.
///
/// Used by the DeviceN /Process /ICCBased path when the embedded
/// process profile is genuinely different from the document
/// OutputIntent profile. The transform flows source CMYK through the
/// source profile's AToB → Lab PCS → destination profile's BToA →
/// destination CMYK, honouring rendering intent and (when configured)
/// Black Point Compensation. The output is the same colour the press
/// would produce if the press were the destination profile.
///
/// Only the `icc-lcms2` backend can construct one of these — qcms 0.3
/// has no CMYK output path. Under the qcms default the constructor
/// returns `None` and `extract_process_paint_cmyk` falls through to
/// the round-5 "natural form" reading. See
/// `HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH` for the full
/// three-state matrix.
pub struct CmykRetargetTransform {
    /// Source and destination profiles, kept for the cache key /
    /// diagnostics surface.
    #[allow(dead_code)]
    src_profile: Arc<IccProfile>,
    #[allow(dead_code)]
    dst_profile: Arc<IccProfile>,
    intent: RenderingIntent,
    inner: <ActiveIccBackend as IccBackend>::CmykRetarget,
}

impl CmykRetargetTransform {
    /// Build a CMYK→CMYK retarget transform. Returns `None` when the
    /// active backend can't compile the transform (no CMYK-out path,
    /// malformed profile bytes, or non-CMYK profiles). The press
    /// default — relative-colorimetric intent + BPC on — is applied;
    /// callers that need a different intent override via
    /// [`Self::new_with_flags`].
    pub fn new(
        src_profile: Arc<IccProfile>,
        dst_profile: Arc<IccProfile>,
        intent: RenderingIntent,
    ) -> Option<Self> {
        Self::new_with_flags(src_profile, dst_profile, intent, TransformFlags::press_default())
    }

    /// Build a CMYK→CMYK retarget transform with explicit flags.
    /// Mainly used by probes that want to pin BPC behaviour
    /// independently of the press default.
    pub fn new_with_flags(
        src_profile: Arc<IccProfile>,
        dst_profile: Arc<IccProfile>,
        intent: RenderingIntent,
        flags: TransformFlags,
    ) -> Option<Self> {
        let inner = <ActiveIccBackend as IccBackend>::build_cmyk_retarget(
            &src_profile,
            &dst_profile,
            intent,
            flags,
        )?;
        Some(Self {
            src_profile,
            dst_profile,
            intent,
            inner,
        })
    }

    /// Retarget a single CMYK quadruple. Inputs and outputs are
    /// unit-interval f32 (channel order C, M, Y, K). The caller is
    /// responsible for any further 8-bit quantisation at the storage
    /// boundary.
    pub fn retarget_pixel(&self, cmyk: [f32; 4]) -> [f32; 4] {
        <ActiveIccBackend as IccBackend>::retarget_cmyk_pixel(&self.inner, cmyk)
    }

    /// The rendering intent the transform was built for.
    pub fn intent(&self) -> RenderingIntent {
        self.intent
    }
}

impl std::fmt::Debug for CmykRetargetTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmykRetargetTransform")
            .field("intent", &self.intent)
            .field("src_bytes", &self.src_profile.bytes.len())
            .field("dst_bytes", &self.dst_profile.bytes.len())
            .field("backend", &backend::active_backend_name())
            .finish()
    }
}

/// Whether the active backend supports CMYK→CMYK retargeting. The
/// gap-closure path in `extract_process_paint_cmyk` consults this to
/// decide between full retargeting and the round-5 "natural form"
/// fallback. Compile-time constant so dead-code elimination keeps the
/// qcms-only build's hot path inlined.
pub const fn active_backend_supports_cmyk_retarget() -> bool {
    cfg!(feature = "icc-lcms2")
}

/// A compiled sRGB → destination-CMYK transform.
///
/// Used by the transparency sidecar's RGB-paint mirror path to convert
/// the rasterised sRGB composite into the document's OutputIntent CMYK
/// space so subsequent transparent CMYK paints over an RGB backdrop
/// composite against the converted backdrop per ISO 32000-1 §11.3.4 +
/// §11.4.5.1 (§11.4.5.1 defines the group's /CS as the single blend
/// colour space; §11.3.4 is the per-pixel compositing computation that
/// runs inside it).
///
/// Only `icc-lcms2` builds construct a real CMM transform. Under
/// `icc-qcms` or no-CMM builds the constructor returns `None`; the
/// call site at `mirror_rgb_paint_into_sidecar` falls through to the
/// §10.3.5 inverse `(C, M, Y) = (1-R, 1-G, 1-B)` with `K = 0`. The
/// fallback loses ink-coverage information in dark areas (no K
/// component) but is colorimetrically sound for the common case where
/// the press recovers K via the same press's GCR/UCR after composition.
pub struct SrgbToCmykTransform {
    /// Destination profile kept for diagnostics + cache key.
    #[allow(dead_code)]
    dst_profile: Arc<IccProfile>,
    intent: RenderingIntent,
    inner: <ActiveIccBackend as IccBackend>::SrgbToCmykTransform,
}

impl SrgbToCmykTransform {
    /// Build an sRGB→destination-CMYK transform using the press
    /// default (relative-colorimetric intent + BPC on). Returns `None`
    /// when the backend can't compile the transform — qcms / no-CMM
    /// builds, or destination profiles that aren't valid CMYK printer
    /// profiles.
    pub fn new(dst_profile: Arc<IccProfile>, intent: RenderingIntent) -> Option<Self> {
        Self::new_with_flags(dst_profile, intent, TransformFlags::press_default())
    }

    /// Build an sRGB→destination-CMYK transform with explicit flags.
    /// The destination profile must declare CMYK by header signature.
    pub fn new_with_flags(
        dst_profile: Arc<IccProfile>,
        intent: RenderingIntent,
        flags: TransformFlags,
    ) -> Option<Self> {
        let inner =
            <ActiveIccBackend as IccBackend>::build_srgb_to_cmyk(&dst_profile, intent, flags)?;
        Some(Self {
            dst_profile,
            intent,
            inner,
        })
    }

    /// Convert a single sRGB pixel to the destination CMYK profile.
    /// Inputs and outputs are unit-interval f32. Caller quantises to
    /// 8-bit at the storage boundary.
    pub fn convert_pixel(&self, rgb: [f32; 3]) -> [f32; 4] {
        <ActiveIccBackend as IccBackend>::convert_srgb_to_cmyk_pixel(&self.inner, rgb)
    }

    /// The rendering intent the transform was built for.
    pub fn intent(&self) -> RenderingIntent {
        self.intent
    }
}

impl std::fmt::Debug for SrgbToCmykTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrgbToCmykTransform")
            .field("intent", &self.intent)
            .field("dst_bytes", &self.dst_profile.bytes.len())
            .field("backend", &backend::active_backend_name())
            .finish()
    }
}

/// Whether the active backend supports sRGB → destination-CMYK
/// conversion through a real CMM transform (vs the §10.3.5 inverse
/// fallback). Compile-time constant so the rendering hot path can be
/// branched at the call site without a runtime check.
pub const fn active_backend_supports_srgb_to_cmyk() -> bool {
    cfg!(feature = "icc-lcms2")
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

    #[test]
    fn active_backend_retarget_capability_matches_feature() {
        let cap = active_backend_supports_cmyk_retarget();
        #[cfg(feature = "icc-lcms2")]
        assert!(cap, "icc-lcms2 build must report retarget capable");
        #[cfg(not(feature = "icc-lcms2"))]
        assert!(!cap, "non-lcms2 build must report retarget UNcapable");
    }
}
