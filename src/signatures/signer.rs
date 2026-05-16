//! PDF signing implementation.
//!
//! This module handles the creation of digital signatures for PDF documents.

use super::byterange::ByteRangeCalculator;
use super::types::{DigestAlgorithm, SignOptions, SigningCredentials};
use crate::error::{Error, Result};

#[cfg(feature = "signatures")]
use sha2::{Digest, Sha256, Sha384, Sha512};

#[cfg(feature = "signatures")]
use sha1::Sha1;

/// PDF signer that creates digital signatures.
pub struct PdfSigner {
    credentials: SigningCredentials,
    options: SignOptions,
    byte_range_calc: ByteRangeCalculator,
}

impl PdfSigner {
    /// Create a new PDF signer with the given credentials and options.
    pub fn new(credentials: SigningCredentials, options: SignOptions) -> Self {
        let byte_range_calc = ByteRangeCalculator::new(options.estimated_size);
        Self {
            credentials,
            options,
            byte_range_calc,
        }
    }

    /// Get the placeholder size for the signature.
    pub fn placeholder_size(&self) -> usize {
        self.byte_range_calc.placeholder_size()
    }

    /// Generate the placeholder for the /Contents value.
    pub fn generate_contents_placeholder(&self) -> String {
        self.byte_range_calc.generate_placeholder()
    }

    /// Build the signature dictionary content (without /Contents value).
    ///
    /// This returns the dictionary entries that should appear in the signature
    /// dictionary. The actual /Contents value should be set to the placeholder.
    pub fn build_signature_dictionary(&self) -> String {
        let mut dict = String::new();

        // Required fields
        dict.push_str("/Type /Sig\n");
        dict.push_str("/Filter /Adobe.PPKLite\n");
        dict.push_str(&format!("/SubFilter /{}\n", self.options.sub_filter.as_pdf_name()));

        // ByteRange placeholder - will be filled in after file is assembled
        dict.push_str("/ByteRange [0 0 0 0]\n");

        // Optional fields
        if let Some(ref name) = self.options.name {
            dict.push_str(&format!("/Name ({})\n", escape_pdf_string(name)));
        }

        if let Some(ref reason) = self.options.reason {
            dict.push_str(&format!("/Reason ({})\n", escape_pdf_string(reason)));
        }

        if let Some(ref location) = self.options.location {
            dict.push_str(&format!("/Location ({})\n", escape_pdf_string(location)));
        }

        if let Some(ref contact) = self.options.contact_info {
            dict.push_str(&format!("/ContactInfo ({})\n", escape_pdf_string(contact)));
        }

        // Signing time (M field)
        let signing_time = format_pdf_date();
        dict.push_str(&format!("/M ({})\n", signing_time));

        dict
    }

    /// Compute the digest of the signed bytes.
    #[cfg(feature = "signatures")]
    pub fn compute_digest(&self, signed_bytes: &[u8]) -> Vec<u8> {
        match self.options.digest_algorithm {
            DigestAlgorithm::Sha1 => {
                let mut hasher = Sha1::new();
                hasher.update(signed_bytes);
                hasher.finalize().to_vec()
            },
            DigestAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(signed_bytes);
                hasher.finalize().to_vec()
            },
            DigestAlgorithm::Sha384 => {
                let mut hasher = Sha384::new();
                hasher.update(signed_bytes);
                hasher.finalize().to_vec()
            },
            DigestAlgorithm::Sha512 => {
                let mut hasher = Sha512::new();
                hasher.update(signed_bytes);
                hasher.finalize().to_vec()
            },
        }
    }

    /// Sign the document and return a DER-encoded CMS/PKCS#7 SignedData blob.
    ///
    /// The returned bytes should be hex-encoded and written to the PDF
    /// `/Contents` placeholder using [`PdfSigner::insert_signature`].
    #[cfg(feature = "signatures")]
    pub fn sign(&self, signed_bytes: &[u8]) -> Result<Vec<u8>> {
        self.create_pkcs7_signature(signed_bytes)
    }

    /// Build a detached CMS SignedData (RFC 5652) over `signed_bytes`.
    ///
    /// Produces a DER-encoded ContentInfo that wraps a SignedData with:
    /// - SHA-256 digest (or the algorithm in `self.options.digest_algorithm`)
    /// - RSA-PKCS#1 v1.5 signature
    /// - Signed attributes: id-contentType + id-messageDigest
    /// - Signer certificate embedded in the certificates field
    ///
    /// The blob is compatible with [`crate::signatures::verify_signer_detached`].
    #[cfg(feature = "signatures")]
    fn create_pkcs7_signature(&self, signed_bytes: &[u8]) -> Result<Vec<u8>> {
        use super::crypto::digest_info_prefix;
        use cms::cert::x509::Certificate as X509Certificate;
        use der::oid::db::rfc5912::{ID_SHA_1, ID_SHA_256, ID_SHA_384, ID_SHA_512};
        use der::{Decode, Encode};
        use rsa::pkcs8::DecodePrivateKey;
        use rsa::{Pkcs1v15Sign, RsaPrivateKey};
        use sha1::Sha1;
        use sha2::{Digest, Sha256, Sha384, Sha512};

        // ── OID byte arrays (pre-encoded, without tag/length) ──────────
        const OID_SIGNED_DATA: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x07, 0x02];
        const OID_DATA: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x07, 0x01];
        const OID_SHA1: &[u8] = &[0x2B, 0x0E, 0x03, 0x02, 0x1A];
        const OID_SHA256: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
        const OID_SHA384: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02];
        const OID_SHA512: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];
        const OID_RSA_ENC: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01];
        const OID_CONTENT_TYPE: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x09, 0x03];
        const OID_MSG_DIGEST: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x09, 0x04];

        // ── Pick digest algorithm ───────────────────────────────────────
        let (digest_oid_bytes, digest_oid, message_digest): (&[u8], _, Vec<u8>) =
            match self.options.digest_algorithm {
                DigestAlgorithm::Sha1 => (OID_SHA1, ID_SHA_1, Sha1::digest(signed_bytes).to_vec()),
                DigestAlgorithm::Sha256 => {
                    (OID_SHA256, ID_SHA_256, Sha256::digest(signed_bytes).to_vec())
                },
                DigestAlgorithm::Sha384 => {
                    (OID_SHA384, ID_SHA_384, Sha384::digest(signed_bytes).to_vec())
                },
                DigestAlgorithm::Sha512 => {
                    (OID_SHA512, ID_SHA_512, Sha512::digest(signed_bytes).to_vec())
                },
            };

        // ── Parse signer certificate (need issuer + serial for SignerInfo) ──
        let cert = X509Certificate::from_der(&self.credentials.certificate)
            .map_err(|e| Error::InvalidPdf(format!("cannot parse signer certificate: {e}")))?;
        let issuer_der = cert
            .tbs_certificate
            .issuer
            .to_der()
            .map_err(|e| Error::InvalidPdf(format!("cannot DER-encode issuer: {e}")))?;
        let serial_der = cert
            .tbs_certificate
            .serial_number
            .to_der()
            .map_err(|e| Error::InvalidPdf(format!("cannot DER-encode serial: {e}")))?;

        // ── Parse RSA private key (PKCS#8, fall back to PKCS#1) ────────
        let rsa_key = RsaPrivateKey::from_pkcs8_der(&self.credentials.private_key)
            .or_else(|_| {
                use pkcs1::DecodeRsaPrivateKey;
                RsaPrivateKey::from_pkcs1_der(&self.credentials.private_key)
            })
            .map_err(|_| {
                Error::InvalidPdf("private key is not valid PKCS#8 or PKCS#1 RSA DER".into())
            })?;

        // ── Signed attributes ──────────────────────────────────────────
        // Attribute 1: id-contentType = id-data
        let attr_ct = {
            let mut c = Vec::new();
            c.extend(der_oid(OID_CONTENT_TYPE));
            c.extend(der_set(&der_oid(OID_DATA)));
            der_sequence(&c)
        };
        // Attribute 2: id-messageDigest = hash of signed_bytes
        let attr_md = {
            let mut c = Vec::new();
            c.extend(der_oid(OID_MSG_DIGEST));
            c.extend(der_set(&der_octet_string(&message_digest)));
            der_sequence(&c)
        };
        // SET OF order: attr_ct < attr_md (shorter encodes first in canonical SET)
        let mut attrs_content = Vec::new();
        attrs_content.extend(&attr_ct);
        attrs_content.extend(&attr_md);

        // For hashing: SET tag (RFC 5652 §5.4)
        let attrs_for_hashing = der_set(&attrs_content);
        // For SignerInfo storage: [0] IMPLICIT replaces the SET tag
        let attrs_for_storage = der_tag(0xA0, &attrs_content);

        // ── Sign: hash(signed_attrs SET) → DigestInfo → RSA sign ───────
        let attrs_hash: Vec<u8> = match self.options.digest_algorithm {
            DigestAlgorithm::Sha1 => Sha1::digest(&attrs_for_hashing).to_vec(),
            DigestAlgorithm::Sha256 => Sha256::digest(&attrs_for_hashing).to_vec(),
            DigestAlgorithm::Sha384 => Sha384::digest(&attrs_for_hashing).to_vec(),
            DigestAlgorithm::Sha512 => Sha512::digest(&attrs_for_hashing).to_vec(),
        };
        let di_prefix = digest_info_prefix(digest_oid)
            .ok_or_else(|| Error::InvalidPdf("no DigestInfo prefix for digest OID".into()))?;
        let mut digest_info_bytes = Vec::with_capacity(di_prefix.len() + attrs_hash.len());
        digest_info_bytes.extend_from_slice(di_prefix);
        digest_info_bytes.extend_from_slice(&attrs_hash);
        let sig_bytes = rsa_key
            .sign(Pkcs1v15Sign::new_unprefixed(), &digest_info_bytes)
            .map_err(|e| Error::InvalidPdf(format!("RSA signing failed: {e}")))?;

        // ── Build SignerInfo ────────────────────────────────────────────
        let signer_info = {
            // IssuerAndSerialNumber SEQUENCE
            let mut isn = Vec::new();
            isn.extend(&issuer_der);
            isn.extend(&serial_der);
            let isn = der_sequence(&isn);

            // digestAlgorithm (no parameters for SHA-*)
            let digest_alg = der_sequence(&der_oid(digest_oid_bytes));

            // signatureAlgorithm: rsaEncryption with NULL params
            let sig_alg = {
                let mut c = Vec::new();
                c.extend(der_oid(OID_RSA_ENC));
                c.extend_from_slice(&[0x05, 0x00]); // NULL
                der_sequence(&c)
            };

            let mut si = Vec::new();
            si.extend(der_integer(1));
            si.extend(isn);
            si.extend(digest_alg);
            si.extend(attrs_for_storage);
            si.extend(sig_alg);
            si.extend(der_octet_string(&sig_bytes));
            der_sequence(&si)
        };

        // ── Build SignedData SEQUENCE ───────────────────────────────────
        let signed_data = {
            // digestAlgorithms SET { SHA-* }
            let digest_algs = der_set(&der_sequence(&der_oid(digest_oid_bytes)));

            // encapContentInfo: id-data, no eContent (detached)
            let encap_ci = der_sequence(&der_oid(OID_DATA));

            // certificates [0] IMPLICIT: the signer cert DER
            let certs = der_tag(0xA0, &self.credentials.certificate);

            // signerInfos SET { signer_info }
            let signer_infos = der_set(&signer_info);

            let mut sd = Vec::new();
            sd.extend(der_integer(1)); // version
            sd.extend(digest_algs);
            sd.extend(encap_ci);
            sd.extend(certs);
            sd.extend(signer_infos);
            der_sequence(&sd)
        };

        // ── Build ContentInfo ───────────────────────────────────────────
        let mut ci = Vec::new();
        ci.extend(der_oid(OID_SIGNED_DATA));
        ci.extend(der_tag(0xA0, &signed_data)); // [0] EXPLICIT wraps SignedData
        Ok(der_sequence(&ci))
    }

    /// Calculate the ByteRange for a prepared PDF.
    pub fn calculate_byte_range(&self, file_size: usize, contents_offset: usize) -> [i64; 4] {
        self.byte_range_calc
            .calculate_byte_range(file_size, contents_offset)
    }

    /// Extract the bytes to be signed from the PDF.
    pub fn extract_signed_bytes(pdf_data: &[u8], byte_range: &[i64; 4]) -> Result<Vec<u8>> {
        ByteRangeCalculator::extract_signed_bytes(pdf_data, byte_range)
    }

    /// Insert the signature into the prepared PDF.
    pub fn insert_signature(
        &self,
        pdf_data: &mut [u8],
        contents_offset: usize,
        signature: &[u8],
    ) -> Result<()> {
        // Convert signature to hex
        let signature_hex = bytes_to_hex(signature);
        self.byte_range_calc
            .insert_signature(pdf_data, contents_offset, &signature_hex)
    }

    /// Get the signing options.
    pub fn options(&self) -> &SignOptions {
        &self.options
    }

    /// Get the signing credentials (certificate info only).
    pub fn credentials(&self) -> &SigningCredentials {
        &self.credentials
    }
}

// ─── Minimal DER / ASN.1 encoding helpers ───────────────────────────────────
//
// These are used only by create_pkcs7_signature. They cover the small subset
// of DER needed for RFC 5652 SignedData: SEQUENCE, SET, OCTET STRING, OID,
// single-byte INTEGER, and arbitrary context-specific constructed tags.

fn der_length(len: usize) -> Vec<u8> {
    if len < 0x80 {
        vec![len as u8]
    } else if len <= 0xFF {
        vec![0x81, len as u8]
    } else if len <= 0xFFFF {
        vec![0x82, (len >> 8) as u8, len as u8]
    } else {
        vec![0x83, (len >> 16) as u8, (len >> 8) as u8, len as u8]
    }
}

fn der_tag(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + content.len());
    out.push(tag);
    out.extend(der_length(content.len()));
    out.extend_from_slice(content);
    out
}

fn der_sequence(content: &[u8]) -> Vec<u8> {
    der_tag(0x30, content)
}
fn der_set(content: &[u8]) -> Vec<u8> {
    der_tag(0x31, content)
}
fn der_oid(oid_bytes: &[u8]) -> Vec<u8> {
    der_tag(0x06, oid_bytes)
}
fn der_octet_string(data: &[u8]) -> Vec<u8> {
    der_tag(0x04, data)
}
fn der_integer(n: u8) -> Vec<u8> {
    vec![0x02, 0x01, n]
}

// ────────────────────────────────────────────────────────────────────────────

/// Convert bytes to uppercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789ABCDEF";
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        hex.push(HEX_CHARS[(byte >> 4) as usize] as char);
        hex.push(HEX_CHARS[(byte & 0x0F) as usize] as char);
    }
    hex
}

/// Escape special characters in a PDF string.
fn escape_pdf_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 10);
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '(' => result.push_str("\\("),
            ')' => result.push_str("\\)"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

/// Current time as a PDF date string. Delegates to the single
/// leap-year-correct implementation; the prior local copy hard-coded
/// month/day to "0101" and approximated the year as 1970 + days/365,
/// corrupting every signature /M date (README latent bug). WASM note:
/// SystemTime::now() in the shared helper still needs cfg-gating if
/// signatures are ever enabled for wasm32 (currently masked).
fn format_pdf_date() -> String {
    super::pdf_date::format_pdf_date_utc()
}

// SignOptions is re-exported from super::types

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_pdf_string() {
        assert_eq!(escape_pdf_string("Hello"), "Hello");
        assert_eq!(escape_pdf_string("Hello (World)"), "Hello \\(World\\)");
        assert_eq!(escape_pdf_string("Line1\nLine2"), "Line1\\nLine2");
        assert_eq!(escape_pdf_string("Path\\to\\file"), "Path\\\\to\\\\file");
    }

    #[test]
    fn test_format_pdf_date() {
        let date = format_pdf_date();
        assert!(date.starts_with("D:"));
        assert!(date.ends_with("Z"));
    }

    #[test]
    fn test_signer_placeholder() {
        let creds = SigningCredentials::new(vec![], vec![]);
        let opts = SignOptions {
            estimated_size: 1024,
            ..Default::default()
        };
        let signer = PdfSigner::new(creds, opts);

        let placeholder = signer.generate_contents_placeholder();
        // 1024 * 2 + 2 = 2050 characters
        assert_eq!(placeholder.len(), 2050);
        assert!(placeholder.starts_with('<'));
        assert!(placeholder.ends_with('>'));
    }

    #[test]
    fn test_build_signature_dictionary() {
        let creds = SigningCredentials::new(vec![], vec![]);
        let opts = SignOptions {
            reason: Some("Test signing".to_string()),
            location: Some("Test City".to_string()),
            ..Default::default()
        };
        let signer = PdfSigner::new(creds, opts);

        let dict = signer.build_signature_dictionary();
        assert!(dict.contains("/Type /Sig"));
        assert!(dict.contains("/Filter /Adobe.PPKLite"));
        assert!(dict.contains("/SubFilter /adbe.pkcs7.detached"));
        assert!(dict.contains("/Reason (Test signing)"));
        assert!(dict.contains("/Location (Test City)"));
        assert!(dict.contains("/ByteRange"));
        assert!(dict.contains("/M (D:"));
    }

    #[test]
    fn test_calculate_byte_range() {
        let creds = SigningCredentials::new(vec![], vec![]);
        let opts = SignOptions {
            estimated_size: 50, // 50 bytes = 102 char placeholder
            ..Default::default()
        };
        let signer = PdfSigner::new(creds, opts);

        let byte_range = signer.calculate_byte_range(1000, 400);
        assert_eq!(byte_range[0], 0);
        assert_eq!(byte_range[1], 400);
        assert_eq!(byte_range[2], 502); // 400 + 102
        assert_eq!(byte_range[3], 498); // 1000 - 502
    }

    #[test]
    #[cfg(feature = "signatures")]
    fn test_sign_produces_valid_cms_blob() {
        use super::super::cms_verify::SignerVerify;
        use super::super::types::SignOptions;
        use super::super::{verify_signer_detached, SigningCredentials};

        let cert_pem = std::fs::read_to_string("tests/fixtures/test_signing_cert.pem")
            .expect("test fixture must exist");
        let key_pem = std::fs::read_to_string("tests/fixtures/test_signing_key.pem")
            .expect("test fixture must exist");
        let creds =
            SigningCredentials::from_pem(&cert_pem, &key_pem).expect("credentials must load");

        let content = b"hello world this is the signed PDF content";
        let signer = PdfSigner::new(creds, SignOptions::default());
        let cms_blob = signer.sign(content).expect("sign must succeed");

        // The produced blob must be parseable and verifiable
        let result = verify_signer_detached(&cms_blob, content)
            .expect("verify_signer_detached must not error");
        assert_eq!(
            result,
            SignerVerify::Valid,
            "signature must verify as Valid with the same content"
        );
    }

    #[test]
    #[cfg(feature = "signatures")]
    fn test_sign_detects_tampered_content() {
        use super::super::cms_verify::SignerVerify;
        use super::super::types::SignOptions;
        use super::super::{verify_signer_detached, SigningCredentials};

        let cert_pem = std::fs::read_to_string("tests/fixtures/test_signing_cert.pem")
            .expect("test fixture must exist");
        let key_pem = std::fs::read_to_string("tests/fixtures/test_signing_key.pem")
            .expect("test fixture must exist");
        let creds =
            SigningCredentials::from_pem(&cert_pem, &key_pem).expect("credentials must load");

        let content = b"original content";
        let tampered = b"tampered content!";
        let signer = PdfSigner::new(creds, SignOptions::default());
        let cms_blob = signer.sign(content).expect("sign must succeed");

        let result = verify_signer_detached(&cms_blob, tampered)
            .expect("verify must not error on tampered content");
        assert_eq!(result, SignerVerify::Invalid, "tampered content must verify as Invalid");
    }

    #[test]
    #[cfg(feature = "signatures")]
    fn test_sign_via_pkcs12() {
        use super::super::cms_verify::SignerVerify;
        use super::super::types::SignOptions;
        use super::super::{verify_signer_detached, SigningCredentials};

        let p12_data =
            std::fs::read("tests/fixtures/test_signing.p12").expect("test fixture must exist");
        let creds =
            SigningCredentials::from_pkcs12(&p12_data, "testpass").expect("PKCS#12 must load");

        let content = b"PDF content for pkcs12 signing test";
        let signer = PdfSigner::new(creds, SignOptions::default());
        let cms_blob = signer.sign(content).expect("sign must succeed");

        let result = verify_signer_detached(&cms_blob, content).expect("verify must not error");
        assert_eq!(result, SignerVerify::Valid, "PKCS#12-signed blob must verify as Valid");
    }
}
