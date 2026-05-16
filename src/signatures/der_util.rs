//! Minimal hand-rolled DER encoders shared by the signature surface
//! (#235 feature plan TODO #2 — DRY: single source of truth).
//!
//! These cover the small subset of DER needed for RFC 5652 SignedData
//! construction (SEQUENCE, SET, OCTET STRING, OID, single-byte INTEGER,
//! arbitrary context-specific constructed tags). They were duplicated in
//! `signer.rs`; promoted here `pub(crate)` so the new PAdES ESS
//! (`pades::ess`), B-T timestamp attribute (`pades::ts_attr`), and DSS
//! writer (`pades::dss`) reuse the *same* tested primitives instead of
//! re-hand-rolling DER (the highest-risk duplication in the crypto path).
//!
//! Definite-length, primitive/constructed BER-DER as used by CMS.

/// DER definite length octets for `len`.
pub(crate) fn der_length(len: usize) -> Vec<u8> {
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

/// `tag || length || content`.
pub(crate) fn der_tag(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + content.len());
    out.push(tag);
    out.extend(der_length(content.len()));
    out.extend_from_slice(content);
    out
}

/// SEQUENCE (tag `0x30`).
pub(crate) fn der_sequence(content: &[u8]) -> Vec<u8> {
    der_tag(0x30, content)
}

/// SET (tag `0x31`).
pub(crate) fn der_set(content: &[u8]) -> Vec<u8> {
    der_tag(0x31, content)
}

/// OBJECT IDENTIFIER (tag `0x06`); `oid_bytes` are the pre-encoded OID
/// content octets (not including tag/length).
pub(crate) fn der_oid(oid_bytes: &[u8]) -> Vec<u8> {
    der_tag(0x06, oid_bytes)
}

/// OCTET STRING (tag `0x04`).
pub(crate) fn der_octet_string(data: &[u8]) -> Vec<u8> {
    der_tag(0x04, data)
}

/// INTEGER (tag `0x02`) for a single unsigned byte (CMS `version` etc.).
pub(crate) fn der_integer(n: u8) -> Vec<u8> {
    vec![0x02, 0x01, n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_short_long_forms() {
        assert_eq!(der_length(0), vec![0x00]);
        assert_eq!(der_length(0x7F), vec![0x7F]);
        assert_eq!(der_length(0x80), vec![0x81, 0x80]);
        assert_eq!(der_length(0xFF), vec![0x81, 0xFF]);
        assert_eq!(der_length(0x0100), vec![0x82, 0x01, 0x00]);
        assert_eq!(der_length(0xFFFF), vec![0x82, 0xFF, 0xFF]);
        assert_eq!(der_length(0x010000), vec![0x83, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn tag_wrappers() {
        assert_eq!(der_sequence(&[0xAA]), vec![0x30, 0x01, 0xAA]);
        assert_eq!(der_set(&[0xBB]), vec![0x31, 0x01, 0xBB]);
        assert_eq!(der_oid(&[0x2A]), vec![0x06, 0x01, 0x2A]);
        assert_eq!(der_octet_string(&[1, 2]), vec![0x04, 0x02, 1, 2]);
        assert_eq!(der_integer(0), vec![0x02, 0x01, 0x00]);
        assert_eq!(der_integer(3), vec![0x02, 0x01, 0x03]);
    }

    #[test]
    fn nested_sequence_length_is_correct() {
        // SEQUENCE { OCTET STRING (130 bytes) } exercises the 0x81 form.
        let inner = der_octet_string(&[0u8; 130]);
        let seq = der_sequence(&inner);
        assert_eq!(seq[0], 0x30);
        // inner len = 1(tag)+2(len 0x81 0x82)+130 = 133 -> 0x81 0x85
        assert_eq!(&seq[1..3], &[0x81, 0x85]);
        assert_eq!(seq.len(), 3 + inner.len());
    }
}
