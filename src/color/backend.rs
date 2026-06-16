//! ICC colour-management backend abstraction.
//!
//! Three CMM backends ship behind feature flags:
//!
//!  - `QcmsBackend` (`icc-qcms`, the default): Firefox's pure-Rust
//!    qcms 0.3 engine. Covers source-profile → sRGB conversion for
//!    every ICC class real PDFs ship (CMYK / RGB / Gray inputs).
//!    Cannot do CMYK → CMYK retargeting (qcms 0.3 has no CMYK output
//!    path) and silently ignores the rendering-intent parameter for
//!    CMYK sources.
//!
//!  - `Lcms2Backend` (`icc-lcms2`, opt-in): Little CMS via the
//!    `lcms2` crate. Press-grade — CMYK→CMYK profile retargeting
//!    through the Lab PCS, Black Point Compensation for relative-
//!    colorimetric (the press default), and rendering-intent dispatch
//!    the spec asks for. Adds a C dependency (`lcms2-sys`).
//!
//!  - `TintboxBackend` (`icc-tintbox`, opt-in): Little CMS reimplemented
//!    in pure Rust (`tintbox`). Same press-grade capability surface as
//!    the lcms2 backend — CMYK→CMYK retargeting, BPC, per-intent
//!    dispatch — with no C dependency, so the full press pipeline stays
//!    available on WASM / C# AOT targets that otherwise fall back to
//!    qcms. tintbox is a byte-for-byte reimplementation of lcms2 on the
//!    unoptimized code path; it differs only in that its lossy curve
//!    optimizer is opt-in (off here) where lcms2's is on by default.
//!
//! At most one backend is *active* per build, selected by capability
//! precedence: tintbox > lcms2 > qcms > none. When several features are
//! enabled the others stay compiled (so the parity differential can
//! drive lcms2 and tintbox side by side) but `ActiveIccBackend` resolves
//! to the highest-precedence one.
//!
//! The [`IccBackend`] trait shape exists so the rest of `crate::color`
//! never imports `qcms` / `lcms2` / `tintbox` directly: every call site
//! goes through [`Transform`](super::Transform) which is built on top of
//! `ActiveIccBackend`. This keeps `color.rs` free of backend cfg gates
//! and confines the per-engine differences to this file.

use super::{IccProfile, RenderingIntent};

/// Transform-construction flags. Mirrors the lcms2 CMM's flag set; the
/// qcms backend reads only the bits it can honour and treats the rest
/// as no-ops.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransformFlags {
    /// Black Point Compensation. The spec doesn't formally require BPC
    /// but the relative-colorimetric press default in real production
    /// pipelines does; without BPC, shadow tones clip to the
    /// destination's black point and the gray balance drifts. lcms2
    /// honours this bit; qcms 0.3 ignores it.
    pub black_point_compensation: bool,
}

impl TransformFlags {
    /// Convenience constructor for the press default — relative-
    /// colorimetric intent with BPC on.
    pub const fn press_default() -> Self {
        Self {
            black_point_compensation: true,
        }
    }
}

/// The trait every ICC backend implements. Two transform classes
/// matter to pdf_oxide:
///
///  - **Source → sRGB** for image / vector composite rendering. Every
///    backend supports it; the qcms 0.3 baseline only supports this.
///  - **CMYK → CMYK retargeting** for DeviceN /Process /ICCBased
///    paints whose embedded profile differs from the document
///    OutputIntent profile. Only lcms2 supports this — qcms 0.3 has
///    no CMYK output side. The retargeting flows through the Lab PCS
///    (CMYK → Lab via source AToB, Lab → CMYK via destination BToA),
///    which is the canonical press path.
///
/// Builders return `None` (rather than panic) when the backend
/// cannot construct a transform for the requested shape. Call sites
/// then fall through to the ISO 32000-1 §10.3.5 additive-clamp
/// formula or the round-5 "natural-form" reading, depending on the
/// context.
pub trait IccBackend {
    /// Backend-specific opaque source-to-sRGB transform handle.
    type SrgbTransform;
    /// Backend-specific opaque CMYK-to-CMYK retargeting transform
    /// handle.
    type CmykRetarget;
    /// Backend-specific opaque sRGB-to-destination-CMYK transform
    /// handle. Used by the transparency sidecar to mirror RGB-source
    /// paints into the CMYK plane so subsequent transparent CMYK
    /// paints composite against the converted backdrop rather than
    /// paper-white per §11.3.4 + §11.4.5.1 (§11.4.5.1 defines the
    /// group's /CS as the single blend colour space; §11.3.4 is the
    /// per-pixel computation that runs inside it).
    type SrgbToCmykTransform;

    /// Build a source-profile → sRGB transform honouring `intent`.
    /// Returns `None` when the backend can't compile the profile
    /// (malformed bytes, unsupported device class, missing tags).
    fn build_srgb_transform(
        profile: &IccProfile,
        intent: RenderingIntent,
        flags: TransformFlags,
    ) -> Option<Self::SrgbTransform>;

    /// Apply a source-to-sRGB transform to one CMYK pixel. Backends
    /// that don't support CMYK source (none currently) should return
    /// `None`. The output is byte-quantised sRGB.
    fn convert_cmyk_pixel(transform: &Self::SrgbTransform, cmyk: [u8; 4]) -> Option<[u8; 3]>;

    /// Apply a source-to-sRGB transform to a packed CMYK buffer.
    /// Output buffer length is `(input.len() / 4) * 3`.
    fn convert_cmyk_buffer(transform: &Self::SrgbTransform, cmyk: &[u8]) -> Option<Vec<u8>>;

    /// Apply a source-to-sRGB transform to a packed RGB buffer.
    /// Output buffer is the same length.
    fn convert_rgb_buffer(transform: &Self::SrgbTransform, rgb: &[u8]) -> Option<Vec<u8>>;

    /// Apply a source-to-sRGB transform to a packed grayscale buffer.
    /// Output buffer is `input.len() * 3` bytes.
    fn convert_gray_buffer(transform: &Self::SrgbTransform, gray: &[u8]) -> Option<Vec<u8>>;

    /// Build a CMYK→CMYK retargeting transform from `src_profile`
    /// (the embedded /ICCBased CMYK profile) to `dst_profile` (the
    /// document `/OutputIntents` CMYK profile) honouring `intent` and
    /// `flags`. Returns `None` when the backend can't do CMYK→CMYK
    /// (the qcms 0.3 baseline) or when profile compilation fails.
    fn build_cmyk_retarget(
        src_profile: &IccProfile,
        dst_profile: &IccProfile,
        intent: RenderingIntent,
        flags: TransformFlags,
    ) -> Option<Self::CmykRetarget>;

    /// Apply a CMYK retargeting transform to a single normalised
    /// CMYK pixel. Inputs and outputs are unit-interval f32 in the
    /// channel order C, M, Y, K. Round-tripping through 8-bit is the
    /// caller's responsibility — the trait operates in f32 so
    /// quantisation only happens at the storage boundary.
    fn retarget_cmyk_pixel(transform: &Self::CmykRetarget, cmyk: [f32; 4]) -> [f32; 4];

    /// Build an sRGB → destination-CMYK transform honouring `intent`
    /// and `flags`. The destination is a printer-class CMYK profile
    /// (typically the document `/OutputIntents` profile). Returns
    /// `None` when the backend can't build the transform — qcms 0.3
    /// has no CMYK output path so it always returns None. lcms2 builds
    /// the transform through sRGB → Lab PCS → destination CMYK.
    fn build_srgb_to_cmyk(
        dst_profile: &IccProfile,
        intent: RenderingIntent,
        flags: TransformFlags,
    ) -> Option<Self::SrgbToCmykTransform>;

    /// Apply an sRGB→destination-CMYK transform to a single sRGB
    /// pixel. Inputs are unit-interval f32 (R, G, B); outputs are
    /// unit-interval f32 (C, M, Y, K). Round-trips through 8-bit at
    /// the lcms2 boundary for the same reason as `retarget_cmyk_pixel`
    /// — the press pipeline serialises plate values as 8-bit.
    fn convert_srgb_to_cmyk_pixel(transform: &Self::SrgbToCmykTransform, rgb: [f32; 3])
        -> [f32; 4];
}

// ============================================================================
// QcmsBackend — pure-Rust default. Mirrors the surface qcms 0.3 exposes.
// ============================================================================

/// qcms-backed [`IccBackend`]. Only the source-to-sRGB methods do real
/// work; CMYK retargeting is unconditionally unsupported in qcms 0.3
/// (no CMYK output path), and that's documented as
/// `HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE_MISMATCH`.
#[cfg(feature = "icc-qcms")]
pub struct QcmsBackend;

#[cfg(feature = "icc-qcms")]
mod qcms_impl {
    use super::*;

    /// Holder so the public trait can stay backend-agnostic. The
    /// inner `qcms::Transform` is the compiled CLUT.
    pub struct SrgbTransform {
        pub(super) inner: qcms::Transform,
        pub(super) source_components: u8,
    }

    /// qcms has no CMYK→CMYK path, so the retarget transform is a
    /// permanent never-constructed marker. We use `core::convert::Infallible`
    /// as the type so it can't be instantiated at runtime — every
    /// `build_cmyk_retarget` call on `QcmsBackend` returns `None`.
    pub struct CmykRetarget(pub(super) core::convert::Infallible);

    /// qcms has no CMYK output path so RGB → CMYK is also unsupported.
    /// `core::convert::Infallible` makes the type uninhabited so the
    /// `convert_srgb_to_cmyk_pixel` arm is unreachable at runtime.
    pub struct SrgbToCmykTransform(pub(super) core::convert::Infallible);

    fn qcms_intent(intent: RenderingIntent) -> qcms::Intent {
        match intent {
            RenderingIntent::Perceptual => qcms::Intent::Perceptual,
            RenderingIntent::RelativeColorimetric => qcms::Intent::RelativeColorimetric,
            RenderingIntent::Saturation => qcms::Intent::Saturation,
            RenderingIntent::AbsoluteColorimetric => qcms::Intent::AbsoluteColorimetric,
        }
    }

    impl IccBackend for QcmsBackend {
        type SrgbTransform = SrgbTransform;
        type CmykRetarget = CmykRetarget;
        type SrgbToCmykTransform = SrgbToCmykTransform;

        fn build_srgb_transform(
            profile: &IccProfile,
            intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::SrgbTransform> {
            let src = qcms::Profile::new_from_slice(profile.bytes(), false)?;
            let dst = qcms::Profile::new_sRGB();
            let src_ty = match profile.n_components() {
                1 => qcms::DataType::Gray8,
                3 => qcms::DataType::RGB8,
                4 => qcms::DataType::CMYK,
                _ => return None,
            };
            qcms::Transform::new_to(&src, &dst, src_ty, qcms::DataType::RGB8, qcms_intent(intent))
                .map(|inner| SrgbTransform {
                    inner,
                    source_components: profile.n_components(),
                })
        }

        fn convert_cmyk_pixel(transform: &Self::SrgbTransform, cmyk: [u8; 4]) -> Option<[u8; 3]> {
            if transform.source_components != 4 {
                return None;
            }
            let mut dst = [0u8; 3];
            transform.inner.convert(&cmyk, &mut dst);
            Some(dst)
        }

        fn convert_cmyk_buffer(transform: &Self::SrgbTransform, cmyk: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 4 {
                return None;
            }
            let pixels = cmyk.len() / 4;
            let mut out = vec![0u8; pixels * 3];
            transform.inner.convert(cmyk, &mut out);
            Some(out)
        }

        fn convert_rgb_buffer(transform: &Self::SrgbTransform, rgb: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 3 {
                return None;
            }
            let mut out = vec![0u8; rgb.len()];
            transform.inner.convert(rgb, &mut out);
            Some(out)
        }

        fn convert_gray_buffer(transform: &Self::SrgbTransform, gray: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 1 {
                return None;
            }
            let mut out = vec![0u8; gray.len() * 3];
            transform.inner.convert(gray, &mut out);
            Some(out)
        }

        fn build_cmyk_retarget(
            _src_profile: &IccProfile,
            _dst_profile: &IccProfile,
            _intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::CmykRetarget> {
            // qcms 0.3 has no CMYK output path. This is the canonical
            // "no" answer that HONEST_GAP_DEVICEN_PROCESS_ICC_PROFILE
            // _MISMATCH documents under the icc-qcms-only build. Call
            // sites fall through to the round-5 "natural form" reading
            // or the §10.3.5 additive-clamp formula.
            None
        }

        fn retarget_cmyk_pixel(transform: &Self::CmykRetarget, _cmyk: [f32; 4]) -> [f32; 4] {
            // Uninhabited: `build_cmyk_retarget` always returns None
            // on QcmsBackend, so this branch is unreachable. We match
            // on the Infallible inhabitant to teach the compiler that.
            match transform.0 {}
        }

        fn build_srgb_to_cmyk(
            _dst_profile: &IccProfile,
            _intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::SrgbToCmykTransform> {
            // qcms 0.3 has no CMYK output path. Call sites fall through
            // to the §10.3.5 inverse `(C, M, Y) = (1-R, 1-G, 1-B)`,
            // `K = 0` formula at the caller.
            None
        }

        fn convert_srgb_to_cmyk_pixel(
            transform: &Self::SrgbToCmykTransform,
            _rgb: [f32; 3],
        ) -> [f32; 4] {
            // Uninhabited under qcms — `build_srgb_to_cmyk` always
            // returns None.
            match transform.0 {}
        }
    }
}

#[cfg(feature = "icc-qcms")]
pub use qcms_impl::{
    CmykRetarget as QcmsCmykRetarget, SrgbToCmykTransform as QcmsSrgbToCmykTransform,
    SrgbTransform as QcmsSrgbTransform,
};

// ============================================================================
// Lcms2Backend — Little CMS via the `lcms2` crate. Press-grade CMM.
// ============================================================================

/// lcms2-backed [`IccBackend`]. Implements the full surface including
/// CMYK→CMYK retargeting (the round-7 gap-closure path) and BPC.
#[cfg(feature = "icc-lcms2")]
pub struct Lcms2Backend;

#[cfg(feature = "icc-lcms2")]
mod lcms2_impl {
    use super::*;

    /// `Transform<u8, u8>` lets us pass `&[u8]` / `&mut [u8]` directly
    /// for every byte-packed pixel format — the lcms2 crate's "u8
    /// special case" handles the reshape internally. PixelFormat
    /// (set in `new_flags`) determines the real channel count.
    /// `DisallowCache` is required (via `Flags::NO_CACHE`) for the
    /// transform to be `Sync` — the per-page IccTransformCache is
    /// shared across rayon worker threads under the `parallel`
    /// feature.
    pub struct SrgbTransform {
        pub(super) inner: lcms2::Transform<u8, u8, lcms2::GlobalContext, lcms2::DisallowCache>,
        pub(super) source_components: u8,
    }

    /// CMYK→CMYK retarget.  Uses `CMYK_8` on both sides (4-channel
    /// byte packed) because lcms2's `CMYK_FLT` encoding treats CMYK
    /// floats as percentages in the 0..100 range — convenient for
    /// ink-coverage UIs, surprising for unit-interval API design.  We
    /// quantise to/from 8-bit at the boundary so the trait surface
    /// can stay in unit-interval f32; the precision loss is bounded
    /// (≤ 1/255) and dominates only when the destination profile's
    /// BToA has sharp transitions — for the prepress / packaging
    /// workloads round 7 targets, 8-bit retarget is the industry-
    /// canonical encoding.  Real production CMMs serialise their CMYK
    /// retargeting LUTs as 8 or 16 bit anyway; floating-point CMYK
    /// PCS handoff is a niche correctness boundary, not the common
    /// case.  `DisallowCache` is required for `Sync` so the
    /// transform can live inside an `Arc` shared across worker
    /// threads under the `parallel` feature.
    pub struct CmykRetarget {
        pub(super) inner:
            lcms2::Transform<[u8; 4], [u8; 4], lcms2::GlobalContext, lcms2::DisallowCache>,
    }

    /// sRGB → destination CMYK. The source is always sRGB (i.e. the
    /// composite pixmap's actual colour space — every RGB-source paint
    /// has been resolved to sRGB by the rasteriser), and the
    /// destination is the document's OutputIntent CMYK profile. The
    /// transform flows sRGB → Lab PCS → destination CMYK so the
    /// §11.3.4 / §11.4.5.1 blend-space conversion happens through the
    /// same canonical PCS path the press uses (§11.4.5.1 is the "ONE
    /// blend space" mandate; §11.3.4 is the per-pixel computation that
    /// runs inside it). Like the `CmykRetarget` above, we quantise at
    /// the 8-bit boundary because press hardware ultimately consumes
    /// 8-bit plates.
    pub struct SrgbToCmykTransform {
        pub(super) inner:
            lcms2::Transform<[u8; 3], [u8; 4], lcms2::GlobalContext, lcms2::DisallowCache>,
    }

    fn lcms2_intent(intent: RenderingIntent) -> lcms2::Intent {
        match intent {
            RenderingIntent::Perceptual => lcms2::Intent::Perceptual,
            RenderingIntent::RelativeColorimetric => lcms2::Intent::RelativeColorimetric,
            RenderingIntent::Saturation => lcms2::Intent::Saturation,
            RenderingIntent::AbsoluteColorimetric => lcms2::Intent::AbsoluteColorimetric,
        }
    }

    fn lcms2_flags(flags: TransformFlags) -> lcms2::Flags<lcms2::DisallowCache> {
        // NO_CACHE is required to make `lcms2::Transform` implement
        // `Sync`.  The pdf_oxide rendering pipeline holds compiled
        // transforms in an `Arc<Transform>` inside the per-page
        // IccTransformCache, and the parallel page-extraction
        // feature shares the same cache across rayon worker threads.
        // The internal 1-pixel cache lcms2 default-enables is a
        // micro-optimisation worth giving up for cross-thread
        // safety; pdf_oxide's per-paint cache already covers the
        // repeat-same-pixel pattern at a coarser grain.
        //
        // BLACKPOINT_COMPENSATION is defined on Flags<AllowCache> in
        // the lcms2 crate, but the `BitOr` impl preserves the cache
        // type of the LHS — so `Flags::NO_CACHE | BPC` produces a
        // `Flags<DisallowCache>` regardless of the BPC constant's
        // declared cache type.
        if flags.black_point_compensation {
            lcms2::Flags::NO_CACHE | lcms2::Flags::BLACKPOINT_COMPENSATION
        } else {
            lcms2::Flags::NO_CACHE
        }
    }

    fn src_pixel_format(n_components: u8) -> Option<lcms2::PixelFormat> {
        match n_components {
            1 => Some(lcms2::PixelFormat::GRAY_8),
            3 => Some(lcms2::PixelFormat::RGB_8),
            4 => Some(lcms2::PixelFormat::CMYK_8),
            _ => None,
        }
    }

    impl IccBackend for Lcms2Backend {
        type SrgbTransform = SrgbTransform;
        type CmykRetarget = CmykRetarget;
        type SrgbToCmykTransform = SrgbToCmykTransform;

        fn build_srgb_transform(
            profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::SrgbTransform> {
            let src = lcms2::Profile::new_icc(profile.bytes()).ok()?;
            let dst = lcms2::Profile::new_srgb();
            let in_fmt = src_pixel_format(profile.n_components())?;
            let out_fmt = lcms2::PixelFormat::RGB_8;
            let inner = lcms2::Transform::new_flags_context(
                lcms2::GlobalContext::new(),
                &src,
                in_fmt,
                &dst,
                out_fmt,
                lcms2_intent(intent),
                lcms2_flags(flags),
            )
            .ok()?;
            Some(SrgbTransform {
                inner,
                source_components: profile.n_components(),
            })
        }

        fn convert_cmyk_pixel(transform: &Self::SrgbTransform, cmyk: [u8; 4]) -> Option<[u8; 3]> {
            if transform.source_components != 4 {
                return None;
            }
            let src: [u8; 4] = cmyk;
            let mut dst = [0u8; 3];
            transform.inner.transform_pixels(&src, &mut dst);
            Some(dst)
        }

        fn convert_cmyk_buffer(transform: &Self::SrgbTransform, cmyk: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 4 {
                return None;
            }
            let pixels = cmyk.len() / 4;
            let mut out = vec![0u8; pixels * 3];
            transform.inner.transform_pixels(cmyk, &mut out);
            Some(out)
        }

        fn convert_rgb_buffer(transform: &Self::SrgbTransform, rgb: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 3 {
                return None;
            }
            let mut out = vec![0u8; rgb.len()];
            transform.inner.transform_pixels(rgb, &mut out);
            Some(out)
        }

        fn convert_gray_buffer(transform: &Self::SrgbTransform, gray: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 1 {
                return None;
            }
            let mut out = vec![0u8; gray.len() * 3];
            transform.inner.transform_pixels(gray, &mut out);
            Some(out)
        }

        fn build_cmyk_retarget(
            src_profile: &IccProfile,
            dst_profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::CmykRetarget> {
            // Both sides must be CMYK by construction. Caller is
            // responsible for that pre-check; we bail anyway if the
            // profile header disagrees.
            if src_profile.n_components() != 4 || dst_profile.n_components() != 4 {
                return None;
            }
            let src = lcms2::Profile::new_icc(src_profile.bytes()).ok()?;
            let dst = lcms2::Profile::new_icc(dst_profile.bytes()).ok()?;
            // Both sides must advertise CmykData — a printer-class
            // profile that secretly emits LabData would otherwise
            // silently produce garbage.
            if !matches!(src.color_space(), lcms2::ColorSpaceSignature::CmykData) {
                return None;
            }
            if !matches!(dst.color_space(), lcms2::ColorSpaceSignature::CmykData) {
                return None;
            }
            let inner = lcms2::Transform::new_flags_context(
                lcms2::GlobalContext::new(),
                &src,
                lcms2::PixelFormat::CMYK_8,
                &dst,
                lcms2::PixelFormat::CMYK_8,
                lcms2_intent(intent),
                lcms2_flags(flags),
            )
            .ok()?;
            Some(CmykRetarget { inner })
        }

        fn retarget_cmyk_pixel(transform: &Self::CmykRetarget, cmyk: [f32; 4]) -> [f32; 4] {
            // Unit-interval f32 in, byte in 0..=255 to lcms2, byte
            // out, then back to unit-interval f32.  The two halves of
            // the round-trip ARE part of the retarget contract: the
            // press hardware ultimately serialises plate values as
            // 8-bit anyway, so an 8-bit clamp at this boundary is the
            // round-trip-faithful encoding.
            let src = [[
                (cmyk[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ]];
            let mut dst = [[0u8; 4]; 1];
            transform.inner.transform_pixels(&src, &mut dst);
            [
                dst[0][0] as f32 / 255.0,
                dst[0][1] as f32 / 255.0,
                dst[0][2] as f32 / 255.0,
                dst[0][3] as f32 / 255.0,
            ]
        }

        fn build_srgb_to_cmyk(
            dst_profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::SrgbToCmykTransform> {
            // Destination must be CMYK by header signature — bail
            // otherwise so callers don't unwittingly write a non-CMYK
            // quadruple into the CMYK sidecar.
            if dst_profile.n_components() != 4 {
                return None;
            }
            let src = lcms2::Profile::new_srgb();
            let dst = lcms2::Profile::new_icc(dst_profile.bytes()).ok()?;
            if !matches!(dst.color_space(), lcms2::ColorSpaceSignature::CmykData) {
                return None;
            }
            let inner = lcms2::Transform::new_flags_context(
                lcms2::GlobalContext::new(),
                &src,
                lcms2::PixelFormat::RGB_8,
                &dst,
                lcms2::PixelFormat::CMYK_8,
                lcms2_intent(intent),
                lcms2_flags(flags),
            )
            .ok()?;
            Some(SrgbToCmykTransform { inner })
        }

        fn convert_srgb_to_cmyk_pixel(
            transform: &Self::SrgbToCmykTransform,
            rgb: [f32; 3],
        ) -> [f32; 4] {
            let src = [[
                (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ]];
            let mut dst = [[0u8; 4]; 1];
            transform.inner.transform_pixels(&src, &mut dst);
            [
                dst[0][0] as f32 / 255.0,
                dst[0][1] as f32 / 255.0,
                dst[0][2] as f32 / 255.0,
                dst[0][3] as f32 / 255.0,
            ]
        }
    }
}

#[cfg(feature = "icc-lcms2")]
pub use lcms2_impl::{
    CmykRetarget as Lcms2CmykRetarget, SrgbToCmykTransform as Lcms2SrgbToCmykTransform,
    SrgbTransform as Lcms2SrgbTransform,
};

// ============================================================================
// TintboxBackend — Little CMS reimplemented in pure Rust (`tintbox`).
// ============================================================================

/// tintbox-backed [`IccBackend`]. Same press-grade capability surface as
/// the `Lcms2Backend` — CMYK→CMYK retargeting through the Lab PCS, Black
/// Point Compensation, and per-intent dispatch — but with no C
/// dependency, so the full press pipeline stays available on the
/// pure-Rust WASM / C# AOT targets that otherwise fall back to qcms.
///
/// tintbox is a byte-for-byte reimplementation of Little CMS, so the
/// compiled transforms match lcms2's on the *unoptimized* code path.
/// The one behavioural difference that matters here: lcms2 enables its
/// lossy curve/matrix optimizer by default, whereas tintbox's optimizer
/// is opt-in (the simple constructors pass `NOOPTIMIZE`). We use the
/// simple constructors, so tintbox takes the accurate path — which is
/// the *more* faithful colour, not a regression.
#[cfg(feature = "icc-tintbox")]
pub struct TintboxBackend;

#[cfg(feature = "icc-tintbox")]
mod tintbox_impl {
    use super::*;
    use tintbox::format::decode::{TYPE_CMYK_8, TYPE_GRAY_8, TYPE_RGB_8};
    use tintbox::opt::OptimizationStrategy;
    use tintbox::profile::{ColorSpace, Profile};
    use tintbox::transform::Transform;

    /// Transforms are built with the `AccurateFast` strategy — tintbox's
    /// LOSSLESS optimizer. It is byte-for-byte identical to the default
    /// `Accurate` path (= lcms2 `-NOOPTIMIZE`; verified by the parity tests)
    /// but routes the CLUT/matrix-shaper eval through a batched path with the
    /// interpolator + input curves resolved once. After tintbox sized its
    /// batched scratch to `min(n, tile)` (so a small-chunk call no longer
    /// allocates a full tile), `AccurateFast` is never slower than `Accurate`
    /// at any chunk size and measured ~25% faster end-to-end on CMYK pages —
    /// so it is the right default here. The lossy `Lcms2Compat` optimizer
    /// stays unused (it would change output bits).
    const TB_STRATEGY: OptimizationStrategy = OptimizationStrategy::AccurateFast;

    /// Holder so the public trait stays backend-agnostic. tintbox's
    /// `Transform` owns its compiled LUT and has no interior mutability,
    /// so it is `Send + Sync` as-is — no cache-disabling flag is needed
    /// to share it in the per-page `Arc<Transform>` across rayon worker
    /// threads (contrast the lcms2 backend, which must pass `NO_CACHE`).
    pub struct SrgbTransform {
        pub(super) inner: Transform,
        pub(super) source_components: u8,
    }

    /// CMYK→CMYK retarget. `TYPE_CMYK_8` on both sides; the unit-interval
    /// f32 trait surface quantises to/from 8-bit at the boundary for the
    /// same reason the lcms2 backend does — press hardware serialises
    /// plates as 8-bit, so the round-trip is faithful, not lossy by
    /// accident.
    pub struct CmykRetarget {
        pub(super) inner: Transform,
    }

    /// sRGB → destination CMYK (the transparency sidecar's RGB→plate
    /// mirror). Source is the sRGB virtual, destination the document
    /// OutputIntent CMYK profile; the link flows sRGB → Lab PCS →
    /// destination CMYK exactly as the lcms2 path does.
    pub struct SrgbToCmykTransform {
        pub(super) inner: Transform,
    }

    fn tintbox_intent(intent: RenderingIntent) -> tintbox::profile::RenderingIntent {
        match intent {
            RenderingIntent::Perceptual => tintbox::profile::RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric => {
                tintbox::profile::RenderingIntent::RelativeColorimetric
            },
            RenderingIntent::Saturation => tintbox::profile::RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric => {
                tintbox::profile::RenderingIntent::AbsoluteColorimetric
            },
        }
    }

    /// The sRGB virtual destination profile — byte-identical to lcms2's
    /// `cmsCreate_sRGBProfile` by tintbox's design, so source→sRGB
    /// transforms compare apples-to-apples with the lcms2 backend.
    fn srgb_profile() -> Option<Profile<'static>> {
        Profile::from_writable(&tintbox::profile::virtuals::build_srgb_profile()).ok()
    }

    fn src_format(n_components: u8) -> Option<u32> {
        match n_components {
            1 => Some(TYPE_GRAY_8),
            3 => Some(TYPE_RGB_8),
            4 => Some(TYPE_CMYK_8),
            _ => None,
        }
    }

    impl IccBackend for TintboxBackend {
        type SrgbTransform = SrgbTransform;
        type CmykRetarget = CmykRetarget;
        type SrgbToCmykTransform = SrgbToCmykTransform;

        fn build_srgb_transform(
            profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::SrgbTransform> {
            let src = Profile::open(profile.bytes()).ok()?;
            let dst = srgb_profile()?;
            let in_fmt = src_format(profile.n_components())?;
            let inner = Transform::new_simple_with_formats_strategy(
                &src,
                &dst,
                tintbox_intent(intent),
                flags.black_point_compensation,
                in_fmt,
                TYPE_RGB_8,
                TB_STRATEGY,
            )
            .ok()?;
            Some(SrgbTransform {
                inner,
                source_components: profile.n_components(),
            })
        }

        fn convert_cmyk_pixel(transform: &Self::SrgbTransform, cmyk: [u8; 4]) -> Option<[u8; 3]> {
            if transform.source_components != 4 {
                return None;
            }
            let mut dst = [0u8; 3];
            transform.inner.do_transform(&cmyk, &mut dst, 1);
            Some(dst)
        }

        fn convert_cmyk_buffer(transform: &Self::SrgbTransform, cmyk: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 4 {
                return None;
            }
            let pixels = cmyk.len() / 4;
            let mut out = vec![0u8; pixels * 3];
            transform.inner.do_transform(cmyk, &mut out, pixels);
            Some(out)
        }

        fn convert_rgb_buffer(transform: &Self::SrgbTransform, rgb: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 3 {
                return None;
            }
            let pixels = rgb.len() / 3;
            let mut out = vec![0u8; rgb.len()];
            transform.inner.do_transform(rgb, &mut out, pixels);
            Some(out)
        }

        fn convert_gray_buffer(transform: &Self::SrgbTransform, gray: &[u8]) -> Option<Vec<u8>> {
            if transform.source_components != 1 {
                return None;
            }
            let pixels = gray.len();
            let mut out = vec![0u8; pixels * 3];
            transform.inner.do_transform(gray, &mut out, pixels);
            Some(out)
        }

        fn build_cmyk_retarget(
            src_profile: &IccProfile,
            dst_profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::CmykRetarget> {
            if src_profile.n_components() != 4 || dst_profile.n_components() != 4 {
                return None;
            }
            let src = Profile::open(src_profile.bytes()).ok()?;
            let dst = Profile::open(dst_profile.bytes()).ok()?;
            // Both sides must advertise CMYK in the header — a printer
            // profile that secretly emits Lab would otherwise silently
            // produce garbage plates.
            if src.header().color_space != ColorSpace::Cmyk
                || dst.header().color_space != ColorSpace::Cmyk
            {
                return None;
            }
            let inner = Transform::new_simple_with_formats_strategy(
                &src,
                &dst,
                tintbox_intent(intent),
                flags.black_point_compensation,
                TYPE_CMYK_8,
                TYPE_CMYK_8,
                TB_STRATEGY,
            )
            .ok()?;
            Some(CmykRetarget { inner })
        }

        fn retarget_cmyk_pixel(transform: &Self::CmykRetarget, cmyk: [f32; 4]) -> [f32; 4] {
            let src = [
                (cmyk[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                (cmyk[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ];
            let mut dst = [0u8; 4];
            transform.inner.do_transform(&src, &mut dst, 1);
            [
                dst[0] as f32 / 255.0,
                dst[1] as f32 / 255.0,
                dst[2] as f32 / 255.0,
                dst[3] as f32 / 255.0,
            ]
        }

        fn build_srgb_to_cmyk(
            dst_profile: &IccProfile,
            intent: RenderingIntent,
            flags: TransformFlags,
        ) -> Option<Self::SrgbToCmykTransform> {
            if dst_profile.n_components() != 4 {
                return None;
            }
            let src = srgb_profile()?;
            let dst = Profile::open(dst_profile.bytes()).ok()?;
            if dst.header().color_space != ColorSpace::Cmyk {
                return None;
            }
            let inner = Transform::new_simple_with_formats_strategy(
                &src,
                &dst,
                tintbox_intent(intent),
                flags.black_point_compensation,
                TYPE_RGB_8,
                TYPE_CMYK_8,
                TB_STRATEGY,
            )
            .ok()?;
            Some(SrgbToCmykTransform { inner })
        }

        fn convert_srgb_to_cmyk_pixel(
            transform: &Self::SrgbToCmykTransform,
            rgb: [f32; 3],
        ) -> [f32; 4] {
            let src = [
                (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ];
            let mut dst = [0u8; 4];
            transform.inner.do_transform(&src, &mut dst, 1);
            [
                dst[0] as f32 / 255.0,
                dst[1] as f32 / 255.0,
                dst[2] as f32 / 255.0,
                dst[3] as f32 / 255.0,
            ]
        }
    }
}

#[cfg(feature = "icc-tintbox")]
pub use tintbox_impl::{
    CmykRetarget as TintboxCmykRetarget, SrgbToCmykTransform as TintboxSrgbToCmykTransform,
    SrgbTransform as TintboxSrgbTransform,
};

// ============================================================================
// NoOpBackend — fallback when no ICC backend feature is enabled.
// ============================================================================

/// No-CMM backend. Every `build_*` returns `None` so call sites in
/// [`crate::color::Transform`] fall straight through to the §10.3.5
/// additive-clamp formula. This is the path WASM / C# AOT consumers
/// hit when they build with `--no-default-features` and don't opt
/// into either ICC feature.
#[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2", feature = "icc-tintbox")))]
pub struct NoOpBackend;

#[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2", feature = "icc-tintbox")))]
mod noop_impl {
    use super::*;

    /// Uninhabited — the `NoOpBackend` never constructs one of these.
    pub struct SrgbTransform(pub(super) core::convert::Infallible);
    /// Uninhabited — the `NoOpBackend` never constructs one of these.
    pub struct CmykRetarget(pub(super) core::convert::Infallible);
    /// Uninhabited — the `NoOpBackend` never constructs one of these.
    pub struct SrgbToCmykTransform(pub(super) core::convert::Infallible);

    impl IccBackend for NoOpBackend {
        type SrgbTransform = SrgbTransform;
        type CmykRetarget = CmykRetarget;
        type SrgbToCmykTransform = SrgbToCmykTransform;

        fn build_srgb_transform(
            _profile: &IccProfile,
            _intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::SrgbTransform> {
            None
        }
        fn convert_cmyk_pixel(transform: &Self::SrgbTransform, _cmyk: [u8; 4]) -> Option<[u8; 3]> {
            match transform.0 {}
        }
        fn convert_cmyk_buffer(transform: &Self::SrgbTransform, _cmyk: &[u8]) -> Option<Vec<u8>> {
            match transform.0 {}
        }
        fn convert_rgb_buffer(transform: &Self::SrgbTransform, _rgb: &[u8]) -> Option<Vec<u8>> {
            match transform.0 {}
        }
        fn convert_gray_buffer(transform: &Self::SrgbTransform, _gray: &[u8]) -> Option<Vec<u8>> {
            match transform.0 {}
        }
        fn build_cmyk_retarget(
            _src_profile: &IccProfile,
            _dst_profile: &IccProfile,
            _intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::CmykRetarget> {
            None
        }
        fn retarget_cmyk_pixel(transform: &Self::CmykRetarget, _cmyk: [f32; 4]) -> [f32; 4] {
            match transform.0 {}
        }
        fn build_srgb_to_cmyk(
            _dst_profile: &IccProfile,
            _intent: RenderingIntent,
            _flags: TransformFlags,
        ) -> Option<Self::SrgbToCmykTransform> {
            None
        }
        fn convert_srgb_to_cmyk_pixel(
            transform: &Self::SrgbToCmykTransform,
            _rgb: [f32; 3],
        ) -> [f32; 4] {
            match transform.0 {}
        }
    }
}

#[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2", feature = "icc-tintbox")))]
pub use noop_impl::{
    CmykRetarget as NoOpCmykRetarget, SrgbToCmykTransform as NoOpSrgbToCmykTransform,
    SrgbTransform as NoOpSrgbTransform,
};

// ============================================================================
// ActiveIccBackend — compile-time selection. lcms2 wins when both are on.
// ============================================================================

// ActiveIccBackend: the backend the rest of `crate::color` dispatches
// through. Resolved at compile time from the feature flag combination
// by capability precedence — tintbox > lcms2 > qcms > none:
//   icc-tintbox                                       → TintboxBackend
//   icc-lcms2 (and not icc-tintbox)                   → Lcms2Backend
//   icc-qcms (and not icc-lcms2 / icc-tintbox)        → QcmsBackend
//   none                                              → NoOpBackend
// tintbox wins over lcms2 because it is the same press-grade capability
// surface with no C dependency; both stay compiled when both features
// are on so the parity differential can drive them side by side.

/// Active ICC backend (compile-time selected — see module docs).
#[cfg(feature = "icc-tintbox")]
pub type ActiveIccBackend = TintboxBackend;

/// Active ICC backend (compile-time selected — see module docs).
#[cfg(all(feature = "icc-lcms2", not(feature = "icc-tintbox")))]
pub type ActiveIccBackend = Lcms2Backend;

/// Active ICC backend (compile-time selected — see module docs).
#[cfg(all(
    feature = "icc-qcms",
    not(feature = "icc-lcms2"),
    not(feature = "icc-tintbox")
))]
pub type ActiveIccBackend = QcmsBackend;

/// Active ICC backend (compile-time selected — see module docs).
#[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2", feature = "icc-tintbox")))]
pub type ActiveIccBackend = NoOpBackend;

/// Backend-name diagnostic for `Debug` output and the
/// `BACKEND_NAME` reporting hook the round-7 probes consume.
pub const fn active_backend_name() -> &'static str {
    #[cfg(feature = "icc-tintbox")]
    {
        "tintbox"
    }
    #[cfg(all(feature = "icc-lcms2", not(feature = "icc-tintbox")))]
    {
        "lcms2"
    }
    #[cfg(all(
        feature = "icc-qcms",
        not(feature = "icc-lcms2"),
        not(feature = "icc-tintbox")
    ))]
    {
        "qcms"
    }
    #[cfg(not(any(feature = "icc-qcms", feature = "icc-lcms2", feature = "icc-tintbox")))]
    {
        "noop"
    }
}

// ============================================================================
// Parity differential — tintbox vs lcms2 (both backends compiled in).
// ============================================================================
//
// These tests run only when BOTH `icc-lcms2` and `icc-tintbox` are enabled, so
// the two engines can be driven side by side over identical real-profile bytes.
// They are the colour-accuracy harness: every test prints its full statistics
// (`cargo test ... -- --nocapture`) before asserting, so a parity break shows
// its magnitude rather than just failing.
//
// Profiles are loaded from the host's standard ColorSync / Adobe profile
// directories. When none are present (e.g. CI on a non-macOS box without the
// Adobe pack) the tests skip with a printed notice rather than fail — committing
// the binary profiles is blocked by their redistribution terms (see
// tests/fixtures/icc/README.md).
#[cfg(all(test, feature = "icc-lcms2", feature = "icc-tintbox"))]
mod tintbox_lcms2_parity {
    use super::*;

    fn read_first(candidates: &[&str]) -> Option<Vec<u8>> {
        candidates.iter().find_map(|p| std::fs::read(p).ok())
    }

    fn srgb_bytes() -> Option<Vec<u8>> {
        read_first(&[
            "/System/Library/ColorSync/Profiles/sRGB Profile.icc",
            "/usr/share/color/icc/sRGB.icc",
        ])
    }

    /// Two distinct real CMYK press profiles for the retarget path.
    fn cmyk_a_bytes() -> Option<Vec<u8>> {
        read_first(&[
            "/Library/Application Support/Adobe/Color/Profiles/Recommended/USWebCoatedSWOP.icc",
            "/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc",
        ])
    }
    fn cmyk_b_bytes() -> Option<Vec<u8>> {
        read_first(&[
            "/Library/Application Support/Adobe/Color/Profiles/Recommended/CoatedFOGRA39.icc",
            "/Library/Application Support/Adobe/Color/Profiles/EuroscaleCoated.icc",
        ])
    }

    fn lcms2_intent() -> lcms2::Intent {
        lcms2::Intent::RelativeColorimetric
    }
    fn tb_intent() -> tintbox::profile::RenderingIntent {
        tintbox::profile::RenderingIntent::RelativeColorimetric
    }

    // ---- lcms2 oracles (byte buffers, BPC on = press default) ----
    fn lcms2_convert(
        src_bytes: &[u8],
        dst_bytes: &[u8],
        in_fmt: lcms2::PixelFormat,
        out_fmt: lcms2::PixelFormat,
        optimize: bool,
        input: &[u8],
        out_channels: usize,
    ) -> Vec<u8> {
        let src = lcms2::Profile::new_icc(src_bytes).unwrap();
        let dst = lcms2::Profile::new_icc(dst_bytes).unwrap();
        // NO_CACHE only disables lcms2's internal 1-pixel memoization; it does
        // not change results, so we omit it here and keep all flags AllowCache
        // (the default Transform cache type) for a single-threaded test.
        let mut flags = lcms2::Flags::BLACKPOINT_COMPENSATION;
        if !optimize {
            flags = flags | lcms2::Flags::NO_OPTIMIZE;
        }
        let t: lcms2::Transform<u8, u8> =
            lcms2::Transform::new_flags(&src, in_fmt, &dst, out_fmt, lcms2_intent(), flags)
                .unwrap();
        let pixels = input.len() / in_fmt_channels(in_fmt);
        let mut out = vec![0u8; pixels * out_channels];
        t.transform_pixels(input, &mut out);
        out
    }

    fn in_fmt_channels(f: lcms2::PixelFormat) -> usize {
        match f {
            lcms2::PixelFormat::GRAY_8 => 1,
            lcms2::PixelFormat::RGB_8 => 3,
            lcms2::PixelFormat::CMYK_8 => 4,
            _ => unreachable!(),
        }
    }

    // ---- tintbox oracles (NOOPTIMIZE via the simple constructor) ----
    fn tb_convert(
        src_bytes: &[u8],
        dst_bytes: Option<&[u8]>,
        in_fmt: u32,
        out_fmt: u32,
        in_channels: usize,
        out_channels: usize,
        input: &[u8],
    ) -> Vec<u8> {
        use tintbox::profile::Profile;
        let src = Profile::open(src_bytes).unwrap();
        let dst = match dst_bytes {
            Some(b) => Profile::open(b).unwrap(),
            None => {
                Profile::from_writable(&tintbox::profile::virtuals::build_srgb_profile()).unwrap()
            },
        };
        // Build with the `AccurateFast` strategy — exactly what the shipped
        // TintboxBackend uses. Asserting this matches lcms2 NO_OPTIMIZE
        // bit-for-bit validates both tintbox's lossless guarantee
        // (AccurateFast == Accurate) and the path pdf_oxide renders through.
        let t = tintbox::transform::Transform::new_simple_with_formats_strategy(
            &src,
            &dst,
            tb_intent(),
            true, // BPC on
            in_fmt,
            out_fmt,
            tintbox::opt::OptimizationStrategy::AccurateFast,
        )
        .unwrap();
        let pixels = input.len() / in_channels;
        let mut out = vec![0u8; pixels * out_channels];
        t.do_transform(input, &mut out, pixels);
        out
    }

    struct Stats {
        n: usize,
        differing: usize,
        max_abs: u16,
        sum_abs: u64,
    }
    fn diff_stats(a: &[u8], b: &[u8]) -> Stats {
        assert_eq!(a.len(), b.len());
        let mut s = Stats {
            n: a.len(),
            differing: 0,
            max_abs: 0,
            sum_abs: 0,
        };
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x as i16 - *y as i16).unsigned_abs();
            if d != 0 {
                s.differing += 1;
            }
            s.max_abs = s.max_abs.max(d);
            s.sum_abs += d as u64;
        }
        s
    }
    fn report(label: &str, s: &Stats) {
        println!(
            "  {label:<46} bytes={:<7} differ={:<7} ({:>5.2}%)  maxΔ={:<3}  meanΔ={:.4}",
            s.n,
            s.differing,
            100.0 * s.differing as f64 / s.n as f64,
            s.max_abs,
            s.sum_abs as f64 / s.n as f64,
        );
    }

    fn cmyk_lattice(step: u16) -> Vec<u8> {
        let vals: Vec<u8> = (0..=255u16)
            .step_by(step as usize)
            .map(|v| v as u8)
            .collect();
        let mut out = Vec::new();
        for &c in &vals {
            for &m in &vals {
                for &y in &vals {
                    for &k in &vals {
                        out.extend_from_slice(&[c, m, y, k]);
                    }
                }
            }
        }
        out
    }
    fn rgb_lattice(step: u16) -> Vec<u8> {
        let vals: Vec<u8> = (0..=255u16)
            .step_by(step as usize)
            .map(|v| v as u8)
            .collect();
        let mut out = Vec::new();
        for &r in &vals {
            for &g in &vals {
                for &b in &vals {
                    out.extend_from_slice(&[r, g, b]);
                }
            }
        }
        out
    }

    #[test]
    fn cmyk_to_srgb_parity_and_optimizer_drift() {
        let (Some(cmyk), Some(srgb)) = (cmyk_a_bytes(), srgb_bytes()) else {
            eprintln!("SKIP cmyk_to_srgb_parity: standard ICC profiles not present");
            return;
        };
        let input = cmyk_lattice(17); // 16^4 = 65536 CMYK pixels
        let tb = tb_convert(
            &cmyk,
            Some(&srgb),
            tintbox::format::decode::TYPE_CMYK_8,
            tintbox::format::decode::TYPE_RGB_8,
            4,
            3,
            &input,
        );
        let lc_noopt = lcms2_convert(
            &cmyk,
            &srgb,
            lcms2::PixelFormat::CMYK_8,
            lcms2::PixelFormat::RGB_8,
            false,
            &input,
            3,
        );
        let lc_opt = lcms2_convert(
            &cmyk,
            &srgb,
            lcms2::PixelFormat::CMYK_8,
            lcms2::PixelFormat::RGB_8,
            true,
            &input,
            3,
        );
        println!("\nCMYK→sRGB (relative+BPC), {} pixels:", input.len() / 4);
        let parity = diff_stats(&tb, &lc_noopt);
        report("tintbox  vs  lcms2(NO_OPTIMIZE)  [parity]", &parity);
        report(
            "lcms2(default) vs lcms2(NO_OPTIMIZE) [optimizer]",
            &diff_stats(&lc_opt, &lc_noopt),
        );
        report("tintbox  vs  lcms2(default)          [as-shipped]", &diff_stats(&tb, &lc_opt));
        assert_eq!(
            parity.max_abs, 0,
            "tintbox must be bit-identical to lcms2 on the unoptimized path"
        );
    }

    #[test]
    fn rgb_to_srgb_parity() {
        let Some(srgb) = srgb_bytes() else {
            eprintln!("SKIP rgb_to_srgb_parity: sRGB profile not present");
            return;
        };
        // Source = a different RGB profile so the transform is non-identity.
        let Some(src_rgb) = read_first(&[
            "/System/Library/ColorSync/Profiles/Generic RGB Profile.icc",
            "/System/Library/ColorSync/Profiles/AdobeRGB1998.icc",
        ]) else {
            eprintln!("SKIP rgb_to_srgb_parity: source RGB profile not present");
            return;
        };
        let input = rgb_lattice(15); // ~18^3
        let tb = tb_convert(
            &src_rgb,
            Some(&srgb),
            tintbox::format::decode::TYPE_RGB_8,
            tintbox::format::decode::TYPE_RGB_8,
            3,
            3,
            &input,
        );
        let lc_noopt = lcms2_convert(
            &src_rgb,
            &srgb,
            lcms2::PixelFormat::RGB_8,
            lcms2::PixelFormat::RGB_8,
            false,
            &input,
            3,
        );
        let lc_opt = lcms2_convert(
            &src_rgb,
            &srgb,
            lcms2::PixelFormat::RGB_8,
            lcms2::PixelFormat::RGB_8,
            true,
            &input,
            3,
        );
        println!("\nRGB→sRGB (relative+BPC), {} pixels:", input.len() / 3);
        let parity = diff_stats(&tb, &lc_noopt);
        report("tintbox  vs  lcms2(NO_OPTIMIZE)  [parity]", &parity);
        report(
            "lcms2(default) vs lcms2(NO_OPTIMIZE) [optimizer]",
            &diff_stats(&lc_opt, &lc_noopt),
        );
        assert_eq!(parity.max_abs, 0, "RGB→sRGB tintbox parity");
    }

    #[test]
    fn cmyk_to_cmyk_retarget_parity() {
        let (Some(a), Some(b)) = (cmyk_a_bytes(), cmyk_b_bytes()) else {
            eprintln!("SKIP cmyk_to_cmyk_retarget_parity: two CMYK profiles not present");
            return;
        };
        let input = cmyk_lattice(17);
        let tb = tb_convert(
            &a,
            Some(&b),
            tintbox::format::decode::TYPE_CMYK_8,
            tintbox::format::decode::TYPE_CMYK_8,
            4,
            4,
            &input,
        );
        let lc_noopt = lcms2_convert(
            &a,
            &b,
            lcms2::PixelFormat::CMYK_8,
            lcms2::PixelFormat::CMYK_8,
            false,
            &input,
            4,
        );
        let lc_opt = lcms2_convert(
            &a,
            &b,
            lcms2::PixelFormat::CMYK_8,
            lcms2::PixelFormat::CMYK_8,
            true,
            &input,
            4,
        );
        println!("\nCMYK→CMYK retarget (relative+BPC), {} pixels:", input.len() / 4);
        let parity = diff_stats(&tb, &lc_noopt);
        report("tintbox  vs  lcms2(NO_OPTIMIZE)  [parity]", &parity);
        report(
            "lcms2(default) vs lcms2(NO_OPTIMIZE) [optimizer]",
            &diff_stats(&lc_opt, &lc_noopt),
        );
        report("tintbox  vs  lcms2(default)          [as-shipped]", &diff_stats(&tb, &lc_opt));
        assert_eq!(parity.max_abs, 0, "CMYK→CMYK retarget tintbox parity");
    }

    #[test]
    fn srgb_to_cmyk_parity() {
        let (Some(srgb), Some(cmyk)) = (srgb_bytes(), cmyk_a_bytes()) else {
            eprintln!("SKIP srgb_to_cmyk_parity: profiles not present");
            return;
        };
        let input = rgb_lattice(15);
        let tb = tb_convert(
            &srgb,
            Some(&cmyk),
            tintbox::format::decode::TYPE_RGB_8,
            tintbox::format::decode::TYPE_CMYK_8,
            3,
            4,
            &input,
        );
        let lc_noopt = lcms2_convert(
            &srgb,
            &cmyk,
            lcms2::PixelFormat::RGB_8,
            lcms2::PixelFormat::CMYK_8,
            false,
            &input,
            4,
        );
        println!("\nsRGB→CMYK (relative+BPC), {} pixels:", input.len() / 3);
        let parity = diff_stats(&tb, &lc_noopt);
        report("tintbox  vs  lcms2(NO_OPTIMIZE)  [parity]", &parity);
        assert_eq!(parity.max_abs, 0, "sRGB→CMYK tintbox parity");
    }

    /// The compiled tintbox transform must be `Send + Sync` so it can live in
    /// the per-page `Arc<Transform>` shared across rayon workers.
    #[test]
    fn tintbox_handles_are_send_sync() {
        fn assert_ss<T: Send + Sync>() {}
        assert_ss::<TintboxSrgbTransform>();
        assert_ss::<TintboxCmykRetarget>();
        assert_ss::<TintboxSrgbToCmykTransform>();
    }

    /// With both features on, tintbox wins the precedence cascade.
    #[test]
    fn active_backend_is_tintbox_when_both_enabled() {
        assert_eq!(active_backend_name(), "tintbox");
    }

    // =====================================================================
    // CMM kernel throughput microbench (perf, not correctness).
    //
    //   cargo test --release --features icc-lcms2,icc-tintbox \
    //       color::backend::tintbox_lcms2_parity::kernel_throughput \
    //       -- --ignored --nocapture --test-threads=1
    //
    // Isolates the transform kernel (no rendering) so a SIMD / interpolation
    // change shows up undiluted by rasterisation + encode. Reports MPx/s for
    // tintbox AccurateFast vs Accurate vs lcms2 (NO_OPTIMIZE and default), on
    // real press profiles, for the two shapes pdf_oxide drives: CMYK→sRGB
    // (image convert) and CMYK→CMYK (process retarget). Build is the only
    // thing that varies between runs, so this is the number to watch when
    // benchmarking a tintbox kernel change in isolation.
    // =====================================================================
    #[test]
    #[ignore = "perf microbench; run with --ignored --release --nocapture"]
    fn kernel_throughput() {
        use std::time::Instant;
        use tintbox::format::decode::{TYPE_CMYK_8, TYPE_RGB_8};
        use tintbox::opt::OptimizationStrategy;
        use tintbox::profile::Profile;

        const N: usize = 2_000_000; // pixels per pass
        const ITERS: usize = 20;

        // Pseudo-varied CMYK so the curves + CLUT do real work (not a constant
        // that would let a cache short-circuit the interpolation).
        let input: Vec<u8> = (0..N * 4)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();

        let mpx = |secs: f64| (N as f64 * ITERS as f64) / secs / 1e6;

        let Some(cmyk) = cmyk_a_bytes() else {
            eprintln!("SKIP kernel_throughput: no CMYK profile present");
            return;
        };

        macro_rules! bench_tb {
            ($label:expr, $dst:expr, $in_fmt:expr, $out_fmt:expr, $out_ch:expr, $strat:expr) => {{
                let src = Profile::open(&cmyk).unwrap();
                let t = tintbox::transform::Transform::new_simple_with_formats_strategy(
                    &src,
                    $dst,
                    tb_intent(),
                    true,
                    $in_fmt,
                    $out_fmt,
                    $strat,
                )
                .unwrap();
                let mut out = vec![0u8; N * $out_ch];
                t.do_transform(&input, &mut out, N); // warm
                let s = Instant::now();
                for _ in 0..ITERS {
                    t.do_transform(&input, &mut out, N);
                }
                println!("  {:38} {:7.2} MPx/s", $label, mpx(s.elapsed().as_secs_f64()));
            }};
        }
        macro_rules! bench_lcms2 {
            ($label:expr, $dst_bytes:expr, $in_pf:expr, $out_pf:expr, $out_ch:expr, $opt:expr) => {{
                let src = lcms2::Profile::new_icc(&cmyk).unwrap();
                let dst = lcms2::Profile::new_icc($dst_bytes).unwrap();
                let mut flags = lcms2::Flags::BLACKPOINT_COMPENSATION;
                if !$opt {
                    flags = flags | lcms2::Flags::NO_OPTIMIZE;
                }
                let t: lcms2::Transform<u8, u8> =
                    lcms2::Transform::new_flags(&src, $in_pf, &dst, $out_pf, lcms2_intent(), flags)
                        .unwrap();
                let mut out = vec![0u8; N * $out_ch];
                t.transform_pixels(&input, &mut out); // warm
                let s = Instant::now();
                for _ in 0..ITERS {
                    t.transform_pixels(&input, &mut out);
                }
                println!("  {:38} {:7.2} MPx/s", $label, mpx(s.elapsed().as_secs_f64()));
            }};
        }

        println!("\nCMM kernel throughput — {N} px × {ITERS} iters (higher = faster)\n");

        if let Some(srgb) = srgb_bytes() {
            let dst = Profile::open(&srgb).unwrap();
            bench_tb!(
                "tintbox CMYK→sRGB AccurateFast",
                &dst,
                TYPE_CMYK_8,
                TYPE_RGB_8,
                3,
                OptimizationStrategy::AccurateFast
            );
            bench_tb!(
                "tintbox CMYK→sRGB Accurate",
                &dst,
                TYPE_CMYK_8,
                TYPE_RGB_8,
                3,
                OptimizationStrategy::Accurate
            );
            bench_lcms2!(
                "lcms2   CMYK→sRGB NO_OPTIMIZE",
                &srgb,
                lcms2::PixelFormat::CMYK_8,
                lcms2::PixelFormat::RGB_8,
                3,
                false
            );
            bench_lcms2!(
                "lcms2   CMYK→sRGB default",
                &srgb,
                lcms2::PixelFormat::CMYK_8,
                lcms2::PixelFormat::RGB_8,
                3,
                true
            );
        }

        if let Some(dstb) = cmyk_b_bytes() {
            println!();
            let dst = Profile::open(&dstb).unwrap();
            bench_tb!(
                "tintbox CMYK→CMYK AccurateFast",
                &dst,
                TYPE_CMYK_8,
                TYPE_CMYK_8,
                4,
                OptimizationStrategy::AccurateFast
            );
            bench_tb!(
                "tintbox CMYK→CMYK Accurate",
                &dst,
                TYPE_CMYK_8,
                TYPE_CMYK_8,
                4,
                OptimizationStrategy::Accurate
            );
            bench_lcms2!(
                "lcms2   CMYK→CMYK NO_OPTIMIZE",
                &dstb,
                lcms2::PixelFormat::CMYK_8,
                lcms2::PixelFormat::CMYK_8,
                4,
                false
            );
            bench_lcms2!(
                "lcms2   CMYK→CMYK default",
                &dstb,
                lcms2::PixelFormat::CMYK_8,
                lcms2::PixelFormat::CMYK_8,
                4,
                true
            );
        }
    }
}
