//! PDF encryption algorithms.
//!
//! This module implements the cryptographic algorithms specified in the PDF specification
//! for key derivation and password validation.
//!
//! PDF Spec: Section 7.6.3 - Standard Security Handler

use md5::{Digest, Md5};

/// Padding string used in PDF encryption (32 bytes).
///
/// PDF Spec: Algorithm 2, step 1
const PADDING: &[u8; 32] = b"\x28\xBF\x4E\x5E\x4E\x75\x8A\x41\
                              \x64\x00\x4E\x56\xFF\xFA\x01\x08\
                              \x2E\x2E\x00\xB6\xD0\x68\x3E\x80\
                              \x2F\x0C\xA9\xFE\x64\x53\x69\x7A";

/// Compute the encryption key from a password (Algorithm 2).
///
/// PDF Spec: Section 7.6.3.3 - Algorithm 2: Computing an encryption key
///
/// # Arguments
///
/// * `password` - User or owner password (up to 32 bytes)
/// * `owner_key` - 32-byte owner password hash from encryption dictionary
/// * `permissions` - User access permissions (P field)
/// * `file_id` - First element of file identifier array
/// * `revision` - Encryption revision number (R field)
/// * `key_length` - Key length in bytes
/// * `encrypt_metadata` - Whether to encrypt metadata
///
/// # Returns
///
/// The derived encryption key
pub fn compute_encryption_key(
    password: &[u8],
    owner_key: &[u8],
    permissions: i32,
    file_id: &[u8],
    revision: u32,
    key_length: usize,
    encrypt_metadata: bool,
) -> Vec<u8> {
    let mut hasher = Md5::new();

    // Step a: Pad or truncate password to 32 bytes
    let mut padded_password = [0u8; 32];
    let pass_len = password.len().min(32);
    padded_password[..pass_len].copy_from_slice(&password[..pass_len]);
    if pass_len < 32 {
        padded_password[pass_len..].copy_from_slice(&PADDING[..(32 - pass_len)]);
    }

    // Step b: Pass the password to MD5
    hasher.update(padded_password);

    // Step c: Pass the owner password hash
    hasher.update(owner_key);

    // Step d: Pass permissions as 32-bit little-endian
    hasher.update(permissions.to_le_bytes());

    // Step e: Pass the file identifier
    hasher.update(file_id);

    // Step f: For R >= 4, if EncryptMetadata is false, pass 0xFFFFFFFF
    if revision >= 4 && !encrypt_metadata {
        hasher.update([0xFF, 0xFF, 0xFF, 0xFF]);
    }

    // Step g: Finish MD5 hash
    let mut hash = hasher.finalize().to_vec();

    // Step h: For R >= 3, do 50 additional MD5 iterations on first key_length bytes
    if revision >= 3 {
        for _ in 0..50 {
            let mut hasher = Md5::new();
            hasher.update(&hash[..key_length]);
            hash = hasher.finalize().to_vec();
        }
    }

    // Step i: Return first key_length bytes
    hash[..key_length].to_vec()
}

/// Pad or truncate a password to 32 bytes using the standard padding.
///
/// PDF Spec: Algorithm 2, step 1
#[allow(dead_code)]
pub fn pad_password(password: &[u8]) -> Vec<u8> {
    let mut padded = Vec::with_capacity(32);
    let pass_len = password.len().min(32);
    padded.extend_from_slice(&password[..pass_len]);
    if pass_len < 32 {
        padded.extend_from_slice(&PADDING[..(32 - pass_len)]);
    }
    padded
}

/// Authenticate the user password (Algorithm 4/5).
///
/// PDF Spec: Section 7.6.3.4 - Algorithm 4/5: User password authentication
///
/// Returns the encryption key if authentication succeeds.
pub fn authenticate_user_password(
    password: &[u8],
    user_key: &[u8],
    owner_key: &[u8],
    permissions: i32,
    file_id: &[u8],
    revision: u32,
    key_length: usize,
    encrypt_metadata: bool,
) -> Option<Vec<u8>> {
    // Compute encryption key from password
    let key = compute_encryption_key(
        password,
        owner_key,
        permissions,
        file_id,
        revision,
        key_length,
        encrypt_metadata,
    );

    // Compute expected user key
    let expected_user_key = if revision >= 3 {
        compute_user_key_r3(&key, file_id)
    } else {
        compute_user_key_r2(&key)
    };

    // Compare first 16 bytes (constant-time comparison)
    let matches = constant_time_compare(&user_key[..16], &expected_user_key[..16]);

    if matches {
        Some(key)
    } else {
        None
    }
}

/// Compute the user password hash for R=2 (Algorithm 4).
///
/// PDF Spec: Section 7.6.3.4 - Algorithm 4
fn compute_user_key_r2(key: &[u8]) -> Vec<u8> {
    // Encrypt padding string with key
    super::rc4::rc4_crypt(key, PADDING)
}

/// Compute the user password hash for R>=3 (Algorithm 5).
///
/// PDF Spec: Section 7.6.3.4 - Algorithm 5
fn compute_user_key_r3(key: &[u8], file_id: &[u8]) -> Vec<u8> {
    // Step a: Create MD5 hash of padding + file ID
    let mut hasher = Md5::new();
    hasher.update(PADDING);
    hasher.update(file_id);
    let mut hash = hasher.finalize().to_vec();

    // Step b: Encrypt the hash 20 times with modified keys
    for i in 0..20 {
        let mut modified_key = key.to_vec();
        for byte in &mut modified_key {
            *byte ^= i as u8;
        }
        hash = super::rc4::rc4_crypt(&modified_key, &hash);
    }

    // Step c: Append 16 arbitrary bytes (we use zeros)
    hash.extend_from_slice(&[0u8; 16]);
    hash
}

/// Constant-time comparison to prevent timing attacks.
///
/// Returns true if the slices are equal.
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_password() {
        let password = b"test";
        let padded = pad_password(password);
        assert_eq!(padded.len(), 32);
        assert_eq!(&padded[..4], b"test");
        assert_eq!(&padded[4..], &PADDING[..28]);
    }

    #[test]
    fn test_pad_password_long() {
        let password = b"this is a very long password that exceeds 32 bytes";
        let padded = pad_password(password);
        assert_eq!(padded.len(), 32);
        assert_eq!(&padded[..], &password[..32]);
    }

    #[test]
    fn test_pad_password_exact() {
        let password = &[0u8; 32];
        let padded = pad_password(password);
        assert_eq!(padded.len(), 32);
        assert_eq!(&padded[..], password);
    }

    #[test]
    fn test_constant_time_compare_equal() {
        let a = b"test1234test1234";
        let b = b"test1234test1234";
        assert!(constant_time_compare(a, b));
    }

    #[test]
    fn test_constant_time_compare_not_equal() {
        let a = b"test1234test1234";
        let b = b"test1234test1235";
        assert!(!constant_time_compare(a, b));
    }

    #[test]
    fn test_constant_time_compare_different_length() {
        let a = b"test";
        let b = b"testing";
        assert!(!constant_time_compare(a, b));
    }

    #[test]
    fn test_compute_encryption_key() {
        let password = b"user";
        let owner_key = &[0u8; 32];
        let permissions = -1;
        let file_id = b"test_file_id";
        let revision = 2;
        let key_length = 5;

        let key = compute_encryption_key(
            password,
            owner_key,
            permissions,
            file_id,
            revision,
            key_length,
            true,
        );

        assert_eq!(key.len(), key_length);
    }
}
