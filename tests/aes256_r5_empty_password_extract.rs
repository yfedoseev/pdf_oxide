//! AES-256 revision 5 (PDF 2.0 / Adobe Extension Level 3): the file
//! encryption key is stored AES-wrapped in /UE — the intermediate key
//! SHA-256(password ‖ key salt) *decrypts* /UE, it is not itself the file
//! key. A reader that skips the unwrap authenticates successfully and then
//! decrypts every stream to noise.
//!
//! The fixture is built here from the spec algorithm using the RustCrypto
//! primitives directly — deliberately not through pdf_oxide's own key
//! derivation, so the fixture cannot inherit the defect under test.

use sha2::{Digest, Sha256};

use pdf_oxide::PdfDocument;

const FILE_KEY: [u8; 32] = [
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
];
const USER_VALIDATION_SALT: [u8; 8] = *b"UVALSLT!";
const USER_KEY_SALT: [u8; 8] = *b"UKEYSLT!";
const OWNER_VALIDATION_SALT: [u8; 8] = *b"OVALSLT!";
const OWNER_KEY_SALT: [u8; 8] = *b"OKEYSLT!";
const CONTENT_IV: [u8; 16] = [0x5A; 16];
const PERMS_P: i32 = -1028;

fn sha256(parts: &[&[u8]]) -> Vec<u8> {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().to_vec()
}

fn aes256_cbc_nopad(key: &[u8], iv: &[u8], data: &[u8]) -> Vec<u8> {
    use aes::cipher::{block_padding::NoPadding, BlockModeEncrypt, KeyIvInit};
    let enc = cbc::Encryptor::<aes::Aes256>::new_from_slices(key, iv).expect("key/iv");
    let mut buf = data.to_vec();
    let len = buf.len();
    enc.encrypt_padded::<NoPadding>(&mut buf, len)
        .expect("encrypt")
        .to_vec()
}

fn aes256_cbc_pkcs7(key: &[u8], iv: &[u8], data: &[u8]) -> Vec<u8> {
    use aes::cipher::{block_padding::Pkcs7, BlockModeEncrypt, KeyIvInit};
    let enc = cbc::Encryptor::<aes::Aes256>::new_from_slices(key, iv).expect("key/iv");
    let mut buf = vec![0u8; data.len() + 16];
    buf[..data.len()].copy_from_slice(data);
    enc.encrypt_padded::<Pkcs7>(&mut buf, data.len())
        .expect("encrypt")
        .to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

/// Build a one-page R5 AES-256 encrypted PDF with empty user and owner
/// passwords, whose page text is "secret payload text".
fn r5_encrypted_pdf() -> Vec<u8> {
    // U: SHA256(pw ‖ validation salt) ‖ validation salt ‖ key salt.
    let mut u_entry = sha256(&[b"", &USER_VALIDATION_SALT]);
    u_entry.extend_from_slice(&USER_VALIDATION_SALT);
    u_entry.extend_from_slice(&USER_KEY_SALT);

    // UE: AES-256-CBC(key = SHA256(pw ‖ key salt), iv = 0) over the file key.
    let ue_entry = aes256_cbc_nopad(&sha256(&[b"", &USER_KEY_SALT]), &[0u8; 16], &FILE_KEY);

    // O: SHA256(pw ‖ owner validation salt ‖ U) ‖ owner validation salt ‖ owner key salt.
    let mut o_entry = sha256(&[b"", &OWNER_VALIDATION_SALT, &u_entry]);
    o_entry.extend_from_slice(&OWNER_VALIDATION_SALT);
    o_entry.extend_from_slice(&OWNER_KEY_SALT);

    // OE: AES-256-CBC(key = SHA256(pw ‖ owner key salt ‖ U), iv = 0) over the file key.
    let oe_entry =
        aes256_cbc_nopad(&sha256(&[b"", &OWNER_KEY_SALT, &u_entry]), &[0u8; 16], &FILE_KEY);

    // Perms: P (little-endian) ‖ FF FF FF FF ‖ 'T' ‖ "adb" ‖ 4 arbitrary bytes,
    // AES-256 encrypted with the file key. A single block with iv = 0 is ECB.
    let mut perms_plain = [0u8; 16];
    perms_plain[..4].copy_from_slice(&PERMS_P.to_le_bytes());
    perms_plain[4..8].copy_from_slice(&[0xFF; 4]);
    perms_plain[8] = b'T';
    perms_plain[9..12].copy_from_slice(b"adb");
    perms_plain[12..16].copy_from_slice(b"salt");
    let perms_entry = aes256_cbc_nopad(&FILE_KEY, &[0u8; 16], &perms_plain);

    // Content stream, AES-256-CBC with the IV prepended (§7.6.2).
    let plaintext: &[u8] = b"BT /F1 12 Tf 72 700 Td (secret payload text) Tj ET";
    let mut stream_data = CONTENT_IV.to_vec();
    stream_data.extend_from_slice(&aes256_cbc_pkcs7(&FILE_KEY, &CONTENT_IV, plaintext));

    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 7]; // ids 1..=6
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    bodies[3] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        .to_vec();
    let mut cs = format!("<< /Length {} >>\nstream\n", stream_data.len()).into_bytes();
    cs.extend_from_slice(&stream_data);
    cs.extend_from_slice(b"\nendstream");
    bodies[4] = cs;
    bodies[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec();
    bodies[6] = format!(
        "<< /Filter /Standard /V 5 /R 5 /Length 256 \
         /CF << /StdCF << /AuthEvent /DocOpen /CFM /AESV3 /Length 32 >> >> \
         /StmF /StdCF /StrF /StdCF /P {PERMS_P} /EncryptMetadata true \
         /O <{}> /U <{}> /OE <{}> /UE <{}> /Perms <{}> >>",
        hex(&o_entry),
        hex(&u_entry),
        hex(&oe_entry),
        hex(&ue_entry),
        hex(&perms_entry),
    )
    .into_bytes();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 7];
    for id in 1..=6 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size 7 /Root 1 0 R /Encrypt 6 0 R \
             /ID [<0123456789ABCDEF0123456789ABCDEF> <0123456789ABCDEF0123456789ABCDEF>] >>\n\
             startxref\n{xref}\n%%EOF\n"
        )
        .as_bytes(),
    );
    out
}

#[test]
fn r5_empty_password_streams_decrypt_to_real_content() {
    let doc = PdfDocument::from_bytes(r5_encrypted_pdf()).expect("parse");
    let authenticated = doc.authenticate(b"").expect("authenticate");
    assert!(authenticated, "empty user password must authenticate on R5");

    let text = doc.extract_text(0).expect("extract_text");
    assert!(
        text.contains("secret payload text"),
        "R5 file key must come from unwrapping /UE; extracted text = {text:?}"
    );
}
