//! #230 — process-wide policy registry: explicit set + set-once.
//!
//! Own integration binary (per-process `OnceLock`). Installs a policy
//! BEFORE any `active_policy()` call, then asserts it is reflected and
//! that a second `set_policy` is rejected (a runtime policy downgrade
//! is an attack vector — set-once like `set_provider`).

use pdf_oxide::crypto::{
    active_policy, set_policy, AlgorithmId, AlgorithmUse, Decision, PolicyMode, SecurityPolicy,
    SetPolicyError,
};

#[test]
fn set_policy_is_reflected_and_set_once() {
    // Install Strict before first access.
    set_policy(SecurityPolicy::strict()).expect("first set_policy must succeed");

    let p = active_policy();
    assert_eq!(p.mode(), PolicyMode::Strict);
    // Strict: legacy readable, weak-write denied.
    assert_eq!(p.evaluate(AlgorithmId::HashMd5, AlgorithmUse::Read), Decision::Allow);
    assert_eq!(p.evaluate(AlgorithmId::HashMd5, AlgorithmUse::Write), Decision::Deny);
    assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Deny);
    assert_eq!(p.evaluate(AlgorithmId::CipherAes256Cbc, AlgorithmUse::Write), Decision::Allow);

    // Set-once: a second attempt is rejected (no mid-flight downgrade).
    let again = set_policy(SecurityPolicy::compat());
    assert!(matches!(again, Err(SetPolicyError::AlreadySet)));
    // Policy unchanged.
    assert_eq!(active_policy().mode(), PolicyMode::Strict);
}
