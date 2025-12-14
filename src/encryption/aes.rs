//! AES encryption/decryption for PDF.
//!
//! AES (Advanced Encryption Standard) is used in PDF 1.6+ for stronger encryption.
//! PDFs use AES in CBC (Cipher Block Chaining) mode with PKCS#7 padding.
//!
//! PDF Spec: Section 7.6.2 - General Encryption Algorithm

use aes::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use aes::Aes128;
use cbc::{Decryptor, Encryptor};

#[allow(dead_code)]
type Aes128CbcEnc = Encryptor<Aes128>;
type Aes128CbcDec = Decryptor<Aes128>;

/// Encrypt data using AES-128 in CBC mode with PKCS#7 padding.
///
/// # Arguments
///
/// * `key` - The 16-byte encryption key
/// * `iv` - The 16-byte initialization vector
/// * `data` - The data to encrypt
///
/// # Returns
///
/// The encrypted data with PKCS#7 padding, or an error if encryption fails
#[allow(dead_code)]
pub fn aes128_encrypt(key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, &'static str> {
    if key.len() != 16 {
        return Err("AES-128 key must be 16 bytes");
    }
    if iv.len() != 16 {
        return Err("IV must be 16 bytes");
    }

    // Apply PKCS#7 padding manually
    let mut padded = data.to_vec();
    let padding_len = 16 - (data.len() % 16);
    padded.extend(std::iter::repeat_n(padding_len as u8, padding_len));

    // Encrypt in-place
    let len = padded.len();
    let cipher = Aes128CbcEnc::new(key.into(), iv.into());
    cipher
        .encrypt_padded_mut::<aes::cipher::block_padding::NoPadding>(&mut padded, len)
        .map_err(|_| "Encryption failed")?;

    Ok(padded)
}

/// Decrypt data using AES-128 in CBC mode and remove PKCS#7 padding.
///
/// # Arguments
///
/// * `key` - The 16-byte encryption key
/// * `iv` - The 16-byte initialization vector
/// * `data` - The encrypted data
///
/// # Returns
///
/// The decrypted data with padding removed, or an error if decryption fails
pub fn aes128_decrypt(key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, &'static str> {
    if key.len() != 16 {
        return Err("AES-128 key must be 16 bytes");
    }
    if iv.len() != 16 {
        return Err("IV must be 16 bytes");
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if !data.len().is_multiple_of(16) {
        return Err("Encrypted data length must be multiple of 16");
    }

    // Decrypt in-place
    let mut buffer = data.to_vec();
    let cipher = Aes128CbcDec::new(key.into(), iv.into());
    let decrypted = cipher
        .decrypt_padded_mut::<aes::cipher::block_padding::NoPadding>(&mut buffer)
        .map_err(|_| "Decryption failed")?;

    // Remove PKCS#7 padding manually
    if decrypted.is_empty() {
        return Ok(Vec::new());
    }

    let padding_len = decrypted[decrypted.len() - 1] as usize;
    if padding_len == 0 || padding_len > 16 {
        return Err("Invalid PKCS#7 padding");
    }

    // Verify padding
    let data_len = decrypted.len().saturating_sub(padding_len);
    for &byte in &decrypted[data_len..] {
        if byte != padding_len as u8 {
            return Err("Invalid PKCS#7 padding");
        }
    }

    Ok(decrypted[..data_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_round_trip() {
        let key = b"0123456789abcdef"; // 16 bytes
        let iv = b"fedcba9876543210"; // 16 bytes
        let plaintext = b"Hello, AES encryption!";

        // Encrypt
        let ciphertext = aes128_encrypt(key, iv, plaintext).unwrap();

        // Decrypt
        let decrypted = aes128_decrypt(key, iv, &ciphertext).unwrap();

        assert_eq!(plaintext, &decrypted[..]);
        assert_ne!(plaintext, &ciphertext[..]);
    }

    #[test]
    fn test_aes128_empty() {
        let key = b"0123456789abcdef";
        let iv = b"fedcba9876543210";
        let plaintext = b"";

        let ciphertext = aes128_encrypt(key, iv, plaintext).unwrap();
        let decrypted = aes128_decrypt(key, iv, &ciphertext).unwrap();

        assert_eq!(decrypted.len(), 0);
    }

    #[test]
    fn test_aes128_block_aligned() {
        let key = b"0123456789abcdef";
        let iv = b"fedcba9876543210";
        let plaintext = b"Exactly16bytes!!"; // 16 bytes

        let ciphertext = aes128_encrypt(key, iv, plaintext).unwrap();
        let decrypted = aes128_decrypt(key, iv, &ciphertext).unwrap();

        assert_eq!(plaintext, &decrypted[..]);
    }

    #[test]
    fn test_aes128_invalid_key() {
        let key = b"short"; // Too short
        let iv = b"fedcba9876543210";
        let plaintext = b"data";

        assert!(aes128_encrypt(key, iv, plaintext).is_err());
    }

    #[test]
    fn test_aes128_different_keys() {
        let iv = b"fedcba9876543210";
        let plaintext = b"Secret message";

        let key1 = b"key1key1key1key1";
        let key2 = b"key2key2key2key2";

        let encrypted1 = aes128_encrypt(key1, iv, plaintext).unwrap();
        let encrypted2 = aes128_encrypt(key2, iv, plaintext).unwrap();

        // Different keys should produce different ciphertexts
        assert_ne!(encrypted1, encrypted2);
    }
}
