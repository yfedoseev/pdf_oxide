//! Error mapping between Rust [`pdf_oxide::Error`] and Java's
//! [`fyi.oxide.pdf.exception.PdfException`] hierarchy.
//!
//! ## Contract (see `docs/releases/plans/v0.3.53/00-common-foundation.md` §5)
//!
//! Every variant in [`pdf_oxide::Error`] maps to exactly one
//! [`PdfErrorKind`] (and thus exactly one Java exception subclass).
//! The mapping is centralised here so JNI entry-points throw the
//! right Java class consistently. CI will eventually fail on any
//! Rust variant that isn't covered (open issue — see v0.3.53 plan
//! `feature-NNN-java-binding.md` DoD axis D).
//!
//! ## Java class names
//!
//! JNI's `FindClass` takes the slash-separated internal binary name
//! (`fyi/oxide/pdf/exception/Foo`), NOT the dot-separated Java name.
//! Constants below are pre-encoded.

use jni::errors::Error as JniError;
use jni::strings::JNIString;
use jni::Env;
use pdf_oxide::Error;

/// Mirror of `fyi.oxide.pdf.exception.PdfErrorKind`.
///
/// We don't expose this to Java directly — the Java side has its
/// own enum. This enum is the single source of truth for "what kind
/// of Java exception do we throw for this Rust error?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfErrorKind {
    Parse,
    Encrypted,
    Permission,
    Io,
    OcrUnavailable,
    Signature,
    InvalidState,
    Unsupported,
    Other,
}

impl PdfErrorKind {
    /// JNI-style binary class name (slashes) for the Java exception
    /// subclass that pairs with this kind.
    pub const fn java_class(self) -> &'static str {
        match self {
            PdfErrorKind::Parse => "fyi/oxide/pdf/exception/PdfParseException",
            PdfErrorKind::Encrypted => "fyi/oxide/pdf/exception/PdfEncryptedException",
            PdfErrorKind::Permission => "fyi/oxide/pdf/exception/PdfPermissionException",
            PdfErrorKind::Io => "fyi/oxide/pdf/exception/PdfIoException",
            PdfErrorKind::OcrUnavailable => "fyi/oxide/pdf/exception/PdfOcrUnavailableException",
            PdfErrorKind::Signature => "fyi/oxide/pdf/exception/PdfSignatureException",
            PdfErrorKind::InvalidState => "fyi/oxide/pdf/exception/PdfInvalidStateException",
            PdfErrorKind::Unsupported => "fyi/oxide/pdf/exception/PdfUnsupportedException",
            PdfErrorKind::Other => "fyi/oxide/pdf/exception/PdfException",
        }
    }
}

/// Map a [`pdf_oxide::Error`] variant to its Java exception kind.
///
/// **This is the canonical mapping for v0.3.53.** Update both here
/// AND the Java side (`PdfErrorKind` enum) when adding new error
/// variants to the Rust core; cross-binding parity tests (DoD axis A)
/// will catch drift.
pub fn classify(err: &Error) -> PdfErrorKind {
    match err {
        // Parse-shaped errors
        Error::InvalidHeader(_)
        | Error::ParseError { .. }
        | Error::ParseWarning { .. }
        | Error::InvalidXref
        | Error::ObjectNotFound(_, _)
        | Error::InvalidObjectType { .. }
        | Error::UnexpectedEof
        | Error::InvalidPdf(_)
        | Error::Decode(_)
        | Error::Font(_)
        | Error::Image(_)
        | Error::CircularReference(_)
        | Error::RecursionLimitExceeded(_)
        | Error::Utf8Error(_) => PdfErrorKind::Parse,

        // I/O failures
        Error::Io(_) => PdfErrorKind::Io,

        // Encryption / authentication
        Error::EncryptedPdf => PdfErrorKind::Encrypted,

        // Unsupported features / formats / versions
        Error::UnsupportedVersion(_) | Error::Unsupported(_) | Error::UnsupportedFilter(_) => {
            PdfErrorKind::Unsupported
        },

        // Operations on handle in a wrong state
        Error::InvalidOperation(_) => PdfErrorKind::InvalidState,

        // Everything else — bucket as OTHER (Encode, Ml, Ocr, LayoutAnalysis,
        // Barcode, and any future variants until classified here).
        _ => PdfErrorKind::Other,
    }
}

/// Throw a Java exception derived from a Rust [`Error`].
///
/// Returns `Err(JniError::JavaException)` on success (per the jni-rs
/// convention — the JVM has now claimed responsibility for
/// propagating the exception, so any Rust code path that follows
/// must short-circuit). Returns a different `Err` only if the
/// `throw_new` JNI call itself failed — which usually means the
/// Java exception class was not packaged into the JAR (a build bug).
pub fn throw_pdf<'local>(env: &mut Env<'local>, err: &Error) -> Result<(), JniError> {
    let kind = classify(err);
    // JNI requires modified-UTF-8 (`JNIStr`/`JNIString`) for both the
    // class binary name and the exception message. `JNIString: From<T>
    // where T: AsRef<str>` does the encoding for us.
    let class = JNIString::from(kind.java_class());
    let msg = JNIString::from(err.to_string());
    env.throw_new(&class, &msg)?;
    Err(JniError::JavaException)
}

/// Throw a `PdfException(kind=OTHER)` carrying the panic payload
/// rendered as a string. Used by JNI entry-points wrapping body
/// closures with [`std::panic::catch_unwind`].
pub fn throw_panic<'local>(
    env: &mut Env<'local>,
    payload: Box<dyn std::any::Any + Send + 'static>,
) -> Result<(), JniError> {
    let msg_string = match payload.downcast::<&'static str>() {
        Ok(s) => format!("panic in JNI shim: {}", *s),
        Err(payload) => match payload.downcast::<String>() {
            Ok(s) => format!("panic in JNI shim: {}", *s),
            Err(_) => "panic in JNI shim (non-string payload)".to_string(),
        },
    };
    let class = JNIString::from(PdfErrorKind::Other.java_class());
    let msg = JNIString::from(msg_string);
    env.throw_new(&class, &msg)?;
    Err(JniError::JavaException)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant of [`PdfErrorKind`] has a Java class name in JNI format.
    /// Format requirement: slash-separated package; ends with `Exception`.
    #[test]
    fn java_class_names_are_well_formed() {
        for kind in [
            PdfErrorKind::Parse,
            PdfErrorKind::Encrypted,
            PdfErrorKind::Permission,
            PdfErrorKind::Io,
            PdfErrorKind::OcrUnavailable,
            PdfErrorKind::Signature,
            PdfErrorKind::InvalidState,
            PdfErrorKind::Unsupported,
            PdfErrorKind::Other,
        ] {
            let cls = kind.java_class();
            assert!(cls.starts_with("fyi/oxide/pdf/exception/"), "kind={:?} class={}", kind, cls);
            assert!(!cls.contains('.'), "JNI class names use slashes, not dots: {}", cls);
            assert!(cls.ends_with("Exception"), "{}", cls);
        }
    }

    /// Spot-check a few of the canonical Rust → Java mappings.
    #[test]
    fn classify_smoke() {
        assert_eq!(classify(&Error::InvalidHeader("X".into())), PdfErrorKind::Parse);
        assert_eq!(classify(&Error::EncryptedPdf), PdfErrorKind::Encrypted);
        assert_eq!(classify(&Error::Unsupported("ZZ".into())), PdfErrorKind::Unsupported);
        assert_eq!(classify(&Error::InvalidOperation("closed".into())), PdfErrorKind::InvalidState);
        let io_err = std::io::Error::other("disk gone");
        assert_eq!(classify(&Error::Io(io_err)), PdfErrorKind::Io);
    }
}
