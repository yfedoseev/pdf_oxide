//! Every file this library writes encrypted must be readable back — by this
//! library, and by anything else following the specification.
//!
//! This is the cheapest high-value test in the project and it did not exist.
//! A writer defect produces a file that *this* library reads back perfectly
//! while another tool rejects it — or, as here, one that nothing can read.
//!
//! **AES-256 (R6).** ISO 32000-2 Algorithm 8 generates one random *file
//! encryption key*, wraps it into `/UE` with a key derived from the user
//! password, and encrypts every stream with that same key. `compute_u_and_ue`
//! produced and returned exactly that key; `EncryptDictBuilder::build`
//! discarded it; and `EncryptionWriteHandler::new` then called
//! `compute_encryption_key`, which for `revision >= 5` returns
//! `generate_random_encryption_key(...)` — **a second, unrelated random key**.
//!
//! So the streams were encrypted under key B while `/UE` wrapped key A. A
//! conforming reader unwraps `/UE`, gets A, and decrypts every stream to
//! noise. The document authenticates first, which is what makes the failure
//! look like a decryption bug rather than a writing one.
//!
//! Aggravating: the specification copy in this repository is ISO 32000-1:2008
//! and contains no R5, R6, `/OE`, `/UE`, `AESV3` or Algorithms 8–13 at all, so
//! no reviewer could have checked this surface against a clause in-tree.

use pdf_oxide::editor::{DocumentEditor, EncryptionAlgorithm, EncryptionConfig, SaveOptions};
use pdf_oxide::writer::{DocumentBuilder, DocumentMetadata, PageSize};
use pdf_oxide::PdfDocument;

const SECRET_TEXT: &str = "ROUND-TRIP-CANARY-7f3a";
const USER_PASSWORD: &str = "user-password";
const OWNER_PASSWORD: &str = "owner-password";

/// A one-page PDF containing `SECRET_TEXT`.
fn plain_pdf() -> Vec<u8> {
    let mut builder = DocumentBuilder::new();
    builder = builder.metadata(DocumentMetadata::new().title("Round Trip"));
    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text(SECRET_TEXT).done();
    }
    builder.build().expect("build plain pdf")
}

/// Write `plain_pdf()` encrypted with `algorithm`.
fn write_encrypted(algorithm: EncryptionAlgorithm) -> Vec<u8> {
    let mut editor = DocumentEditor::from_bytes(plain_pdf()).expect("open plain pdf");
    let config = EncryptionConfig::new(USER_PASSWORD, OWNER_PASSWORD).with_algorithm(algorithm);
    editor
        .save_to_bytes_with_options(SaveOptions::with_encryption(config))
        .expect("save encrypted pdf")
}

/// Write encrypted, then read back **in the same process** with the same
/// password and recover the text.
fn round_trip(algorithm: EncryptionAlgorithm, password: &str) -> String {
    let bytes = write_encrypted(algorithm);
    let doc = PdfDocument::from_bytes(bytes).expect("parse the file we just wrote");
    assert!(doc.is_encrypted(), "the file we wrote must be encrypted");
    assert!(
        doc.authenticate(password.as_bytes()).expect("authenticate"),
        "the password we encrypted with must authenticate"
    );
    doc.extract_text(0)
        .expect("extract text from our own output")
}

/// The gate. Encrypt, read back, compare.
#[test]
fn aes_256_survives_a_write_read_round_trip() {
    let text = round_trip(EncryptionAlgorithm::Aes256, USER_PASSWORD);
    assert!(
        text.contains(SECRET_TEXT),
        "AES-256 output did not decrypt back to its own content — got {text:?}"
    );
}

/// The owner password must reach the same content.
#[test]
fn aes_256_round_trips_under_the_owner_password() {
    let text = round_trip(EncryptionAlgorithm::Aes256, OWNER_PASSWORD);
    assert!(
        text.contains(SECRET_TEXT),
        "AES-256 output did not decrypt under the owner password — got {text:?}"
    );
}

/// The neighbouring revisions, so a fix cannot move the defect sideways.
///
/// Both are unavailable under `fips`: RC4 is not an approved cipher and the
/// R4 key derivation is built on MD5, which FIPS 140-3 forbids. The crate
/// makes that a build-time exclusion, so the two revisions simply do not
/// exist in a FIPS build and there is nothing here to round-trip. AES-256
/// above is approved and runs in every configuration.
#[cfg(not(feature = "fips"))]
#[test]
fn aes_128_survives_a_write_read_round_trip() {
    let text = round_trip(EncryptionAlgorithm::Aes128, USER_PASSWORD);
    assert!(text.contains(SECRET_TEXT), "AES-128 round trip failed — got {text:?}");
}

#[cfg(not(feature = "fips"))]
#[test]
fn rc4_128_survives_a_write_read_round_trip() {
    let text = round_trip(EncryptionAlgorithm::Rc4_128, USER_PASSWORD);
    assert!(text.contains(SECRET_TEXT), "RC4-128 round trip failed — got {text:?}");
}

/// Encryption must actually be applied: the canary must not be sitting in the
/// output as plaintext. Without this, a "round trip" would also pass on a file
/// that was never encrypted at all.
#[test]
fn test_written_file_is_actually_encrypted() {
    // A FIPS build offers only the approved revision; see above.
    #[cfg(not(feature = "fips"))]
    let algorithms = [
        EncryptionAlgorithm::Aes256,
        EncryptionAlgorithm::Aes128,
        EncryptionAlgorithm::Rc4_128,
    ];
    #[cfg(feature = "fips")]
    let algorithms = [EncryptionAlgorithm::Aes256];

    for algorithm in algorithms {
        let bytes = write_encrypted(algorithm);
        let needle = SECRET_TEXT.as_bytes();
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(!found, "{algorithm:?} left the canary in the output as plaintext");
    }
}

/// A wrong password must not authenticate — the round trip must not be
/// passing because authentication is a no-op.
#[test]
fn test_wrong_password_does_not_authenticate() {
    let bytes = write_encrypted(EncryptionAlgorithm::Aes256);
    let doc = PdfDocument::from_bytes(bytes).expect("parse");
    assert!(
        !doc.authenticate(b"not-the-password").expect("authenticate"),
        "a wrong password must be rejected"
    );
}
