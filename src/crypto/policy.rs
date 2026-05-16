//! Runtime cryptographic algorithm-governance policy (issue #230).
//!
//! PR #465 / v0.3.44 (issue #236) shipped *algorithm isolation*: every
//! PDF crypto operation routes through [`crate::crypto::active`]. v0.3.45
//! added the `legacy-crypto` *compile-time* gate. This module adds the
//! missing piece: a **runtime, configurable security policy** — the
//! ability to say "this process may *read* RC4 PDFs but must *never
//! write* anything weaker than AES-256", or "deny SHA-1 signing",
//! without recompiling and without writing a custom
//! [`CryptoProvider`](super::CryptoProvider).
//!
//! This file is the self-contained value-type core (plan §3–4, tasks
//! #1–#4): [`SecurityPolicy`], [`PolicyMode`], [`Decision`],
//! [`AlgorithmId`], [`AlgorithmUse`], the mode-default matrix, the
//! decision precedence, the `FromStr`/`Display` grammar, and the audit
//! seam. The process-wide registry and the `PolicyEnforcedProvider`
//! decorator compose on top of these types in a later increment.
//!
//! Design (SOLID): the policy is **orthogonal to the provider** —
//! mirrors OpenSSL 3 `update-crypto-policies` and .NET
//! `CryptoConfig.AllowOnlyFipsAlgorithms`. The policy never widens
//! behaviour, only narrows it (Liskov-safe when used by the future
//! decorator).
//!
//! Fail-closed is the central invariant: an unknown algorithm, an
//! unparseable spec, or any ambiguity resolves to **deny**, never a
//! silent allow.

use std::collections::BTreeMap;
use std::str::FromStr;

use super::error::AlgorithmKind;

/// A cryptographic primitive `pdf_oxide` can be asked to perform.
///
/// `#[non_exhaustive]` so post-quantum ids (`MlKem*`, `MlDsa*`,
/// `SlhDsa*`) are an additive, non-breaking future change (CNSA 2.0
/// roadmap — see the #230 plan).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AlgorithmId {
    /// MD5 hash (legacy; PDF Standard Security R≤4 password KDF).
    HashMd5,
    /// SHA-1 hash (legacy; historical signatures, `adbe.pkcs7.sha1`).
    HashSha1,
    /// SHA-256 hash (FIPS approved).
    HashSha256,
    /// SHA-384 hash (FIPS approved).
    HashSha384,
    /// SHA-512 hash (FIPS approved).
    HashSha512,
    /// RC4 stream cipher (legacy; PDF R≤4).
    CipherRc4,
    /// AES-128-CBC (PDF V=4 R=4, `AESV2`).
    CipherAes128Cbc,
    /// AES-256-CBC (PDF V=5 R=5/6, `AESV3`).
    CipherAes256Cbc,
    /// RSA PKCS#1 v1.5 signature over SHA-1 (legacy).
    SigRsaPkcs1v15Sha1,
    /// RSA PKCS#1 v1.5 signature over SHA-256.
    SigRsaPkcs1v15Sha256,
    /// RSA PKCS#1 v1.5 signature over SHA-384.
    SigRsaPkcs1v15Sha384,
    /// RSA PKCS#1 v1.5 signature over SHA-512.
    SigRsaPkcs1v15Sha512,
    /// RSA-PSS signature over SHA-256.
    SigRsaPssSha256,
    /// RSA-PSS signature over SHA-384.
    SigRsaPssSha384,
    /// RSA-PSS signature over SHA-512.
    SigRsaPssSha512,
    /// ECDSA P-256 signature over SHA-256.
    SigEcdsaP256Sha256,
    /// ECDSA P-384 signature over SHA-384.
    SigEcdsaP384Sha384,
}

impl AlgorithmId {
    /// Every algorithm id this build knows, in declaration order.
    /// Used by the policy-matrix tests and `inventory()`.
    pub const ALL: [AlgorithmId; 17] = [
        AlgorithmId::HashMd5,
        AlgorithmId::HashSha1,
        AlgorithmId::HashSha256,
        AlgorithmId::HashSha384,
        AlgorithmId::HashSha512,
        AlgorithmId::CipherRc4,
        AlgorithmId::CipherAes128Cbc,
        AlgorithmId::CipherAes256Cbc,
        AlgorithmId::SigRsaPkcs1v15Sha1,
        AlgorithmId::SigRsaPkcs1v15Sha256,
        AlgorithmId::SigRsaPkcs1v15Sha384,
        AlgorithmId::SigRsaPkcs1v15Sha512,
        AlgorithmId::SigRsaPssSha256,
        AlgorithmId::SigRsaPssSha384,
        AlgorithmId::SigRsaPssSha512,
        AlgorithmId::SigEcdsaP256Sha256,
        AlgorithmId::SigEcdsaP384Sha384,
    ];

    /// Stable lowercase token used in the policy grammar, audit logs,
    /// and binding strings. Round-trips with [`Self::from_token`].
    pub const fn token(self) -> &'static str {
        match self {
            AlgorithmId::HashMd5 => "md5",
            AlgorithmId::HashSha1 => "sha1",
            AlgorithmId::HashSha256 => "sha256",
            AlgorithmId::HashSha384 => "sha384",
            AlgorithmId::HashSha512 => "sha512",
            AlgorithmId::CipherRc4 => "rc4",
            AlgorithmId::CipherAes128Cbc => "aes128",
            AlgorithmId::CipherAes256Cbc => "aes256",
            AlgorithmId::SigRsaPkcs1v15Sha1 => "rsa-pkcs1-sha1",
            AlgorithmId::SigRsaPkcs1v15Sha256 => "rsa-pkcs1-sha256",
            AlgorithmId::SigRsaPkcs1v15Sha384 => "rsa-pkcs1-sha384",
            AlgorithmId::SigRsaPkcs1v15Sha512 => "rsa-pkcs1-sha512",
            AlgorithmId::SigRsaPssSha256 => "rsa-pss-sha256",
            AlgorithmId::SigRsaPssSha384 => "rsa-pss-sha384",
            AlgorithmId::SigRsaPssSha512 => "rsa-pss-sha512",
            AlgorithmId::SigEcdsaP256Sha256 => "ecdsa-p256-sha256",
            AlgorithmId::SigEcdsaP384Sha384 => "ecdsa-p384-sha384",
        }
    }

    /// Parse a grammar/binding token back to an id. `None` for an
    /// unknown token (the caller treats that as a fail-closed parse
    /// error — never a silent allow).
    pub fn from_token(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.token() == s)
    }

    /// The broad algorithm family, reusing the existing
    /// [`AlgorithmKind`] so error/audit grouping stays DRY. Signature
    /// ids report `SignatureSign` as their family; the sign-vs-verify
    /// distinction is the orthogonal [`AlgorithmUse`] axis.
    pub const fn kind(self) -> AlgorithmKind {
        match self {
            AlgorithmId::HashMd5
            | AlgorithmId::HashSha1
            | AlgorithmId::HashSha256
            | AlgorithmId::HashSha384
            | AlgorithmId::HashSha512 => AlgorithmKind::Hash,
            AlgorithmId::CipherRc4
            | AlgorithmId::CipherAes128Cbc
            | AlgorithmId::CipherAes256Cbc => AlgorithmKind::SymmetricCipher,
            AlgorithmId::SigRsaPkcs1v15Sha1
            | AlgorithmId::SigRsaPkcs1v15Sha256
            | AlgorithmId::SigRsaPkcs1v15Sha384
            | AlgorithmId::SigRsaPkcs1v15Sha512
            | AlgorithmId::SigRsaPssSha256
            | AlgorithmId::SigRsaPssSha384
            | AlgorithmId::SigRsaPssSha512
            | AlgorithmId::SigEcdsaP256Sha256
            | AlgorithmId::SigEcdsaP384Sha384 => AlgorithmKind::SignatureSign,
        }
    }

    /// Whether this primitive is FIPS 140-3 / SP 800-131A approved for
    /// **new** use. MD5, SHA-1, RC4 and RSA-PKCS#1-v1.5-over-SHA-1 are
    /// not. SHA-1 *verification* of historical signatures is permitted
    /// by SP 800-131A — that exception is modelled in the mode matrix
    /// via [`AlgorithmUse::Read`], not here.
    pub const fn is_fips_approved(self) -> bool {
        matches!(
            self,
            AlgorithmId::HashSha256
                | AlgorithmId::HashSha384
                | AlgorithmId::HashSha512
                | AlgorithmId::CipherAes128Cbc
                | AlgorithmId::CipherAes256Cbc
                | AlgorithmId::SigRsaPkcs1v15Sha256
                | AlgorithmId::SigRsaPkcs1v15Sha384
                | AlgorithmId::SigRsaPkcs1v15Sha512
                | AlgorithmId::SigRsaPssSha256
                | AlgorithmId::SigRsaPssSha384
                | AlgorithmId::SigRsaPssSha512
                | AlgorithmId::SigEcdsaP256Sha256
                | AlgorithmId::SigEcdsaP384Sha384
        )
    }

    /// Coarse security strength in bits (NIST SP 800-57 ballpark).
    /// Used only by the `min_security_bits` Write floor — deliberately
    /// coarse for v0.3.50 (per-algorithm floors are roadmap).
    pub const fn min_security_bits(self) -> u16 {
        match self {
            AlgorithmId::HashMd5 | AlgorithmId::HashSha1 | AlgorithmId::CipherRc4 => 0,
            AlgorithmId::SigRsaPkcs1v15Sha1 => 0,
            AlgorithmId::HashSha256
            | AlgorithmId::CipherAes128Cbc
            | AlgorithmId::SigRsaPkcs1v15Sha256
            | AlgorithmId::SigRsaPssSha256
            | AlgorithmId::SigEcdsaP256Sha256 => 128,
            AlgorithmId::HashSha384
            | AlgorithmId::SigRsaPkcs1v15Sha384
            | AlgorithmId::SigRsaPssSha384
            | AlgorithmId::SigEcdsaP384Sha384 => 192,
            AlgorithmId::HashSha512
            | AlgorithmId::CipherAes256Cbc
            | AlgorithmId::SigRsaPkcs1v15Sha512
            | AlgorithmId::SigRsaPssSha512 => 256,
        }
    }

    /// True for the SHA-1 hash specifically — the one primitive
    /// `FipsStrict` still permits for *read* (historical-signature
    /// verification, NIST SP 800-131A).
    const fn is_sha1_hash(self) -> bool {
        matches!(self, AlgorithmId::HashSha1)
    }
}

/// Direction of a cryptographic operation. The spec forces an
/// asymmetry: a regulated operator must be able to *read* a legacy
/// RC4 PDF (incident response, archival) while being *forbidden* to
/// *produce* anything that weak.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AlgorithmUse {
    /// Decrypt / verify an existing signature / parse legacy material.
    Read,
    /// Encrypt-on-save / new key derivation / new signature creation.
    Write,
}

impl AlgorithmUse {
    /// Grammar token (`"read"` / `"write"`).
    pub const fn token(self) -> &'static str {
        match self {
            AlgorithmUse::Read => "read",
            AlgorithmUse::Write => "write",
        }
    }

    /// Parse a grammar token; `None` is a fail-closed parse error.
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "read" => Some(AlgorithmUse::Read),
            "write" => Some(AlgorithmUse::Write),
            _ => None,
        }
    }
}

/// The outcome of a policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The operation is permitted.
    Allow,
    /// The operation is forbidden by policy (fail-closed default).
    Deny,
}

/// A first-class policy preset.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    /// Today's behaviour. Allows everything the active provider +
    /// `legacy-crypto` allow. Read & write of R≤4 OK. Default.
    Compat,
    /// Read legacy OK; deny weak crypto for *writes* and new
    /// signatures. New encryption must be FIPS-grade.
    Strict,
    /// Deny every non-FIPS-approved algorithm for both directions.
    /// Pairs with `--features fips`. SHA-1 *verify* of historical
    /// signatures still allowed (SP 800-131A).
    FipsStrict,
}

impl PolicyMode {
    /// Grammar token.
    pub const fn token(self) -> &'static str {
        match self {
            PolicyMode::Compat => "compat",
            PolicyMode::Strict => "strict",
            PolicyMode::FipsStrict => "fips-strict",
        }
    }

    /// Default `min_security_bits` Write floor for this mode.
    const fn default_min_bits(self) -> u16 {
        match self {
            PolicyMode::Compat => 0,
            PolicyMode::Strict | PolicyMode::FipsStrict => 128,
        }
    }

    /// The mode's built-in decision for `(alg, use_)`, *before*
    /// explicit overrides and the `min_security_bits` floor. This is
    /// the §3.4 matrix encoded verbatim.
    fn default_decision(self, alg: AlgorithmId, use_: AlgorithmUse) -> Decision {
        match self {
            // Compat allows everything; the compile-time `legacy-crypto`
            // gate and the provider are the remaining guards.
            PolicyMode::Compat => Decision::Allow,
            PolicyMode::Strict => match use_ {
                // Read legacy OK.
                AlgorithmUse::Read => Decision::Allow,
                // Write: only FIPS-approved primitives.
                AlgorithmUse::Write => {
                    if alg.is_fips_approved() {
                        Decision::Allow
                    } else {
                        Decision::Deny
                    }
                },
            },
            PolicyMode::FipsStrict => {
                if alg.is_fips_approved() {
                    Decision::Allow
                } else {
                    match use_ {
                        // Historical-signature verification: SHA-1
                        // hash only (NIST SP 800-131A).
                        AlgorithmUse::Read if alg.is_sha1_hash() => Decision::Allow,
                        _ => Decision::Deny,
                    }
                }
            },
        }
    }
}

/// Error parsing a [`SecurityPolicy`] from its string grammar.
/// Surfaced to the caller; the caller must treat a parse failure as
/// fatal (fail-closed — the policy is *not* installed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyParseError(String);

impl PolicyParseError {
    /// The human-readable parse failure detail.
    pub fn detail(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PolicyParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid crypto policy spec: {}", self.0)
    }
}

impl std::error::Error for PolicyParseError {}

/// A runtime cryptographic governance policy.
///
/// Construct via [`Self::compat`], [`Self::strict`],
/// [`Self::fips_strict`], the [`SecurityPolicyBuilder`], or
/// [`str::parse`]. Evaluate a `(AlgorithmId, AlgorithmUse)` pair with
/// [`Self::evaluate`].
///
/// Decision precedence (documented and tested):
/// 1. An explicit `deny` override is **terminal** (fail-closed).
/// 2. An explicit `allow` override wins over the mode default —
///    *except* `FipsStrict` ignores `allow` overrides for
///    non-FIPS-approved primitives (the .NET `AllowOnlyFipsAlgorithms`
///    precedent).
/// 3. Otherwise the [`PolicyMode`] default matrix applies, then the
///    `min_security_bits` Write floor.
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    mode: PolicyMode,
    overrides: BTreeMap<(AlgorithmId, AlgorithmUse), Decision>,
    /// Decision for an algorithm id this build does not know. Always
    /// `Deny` in v0.3.50 (fail-closed); a field so the forward-shaped
    /// API can relax it deliberately later.
    unknown_algorithm: Decision,
    min_security_bits: u16,
}

impl SecurityPolicy {
    /// The default, behaviour-preserving policy (`Compat`). Allows
    /// everything the provider + compile gates allow — byte-for-byte
    /// identical to pre-policy behaviour.
    pub fn compat() -> Self {
        Self::builder(PolicyMode::Compat).build()
    }

    /// `Strict`: read legacy OK, deny weak crypto on write / new
    /// signatures.
    pub fn strict() -> Self {
        Self::builder(PolicyMode::Strict).build()
    }

    /// `FipsStrict`: deny all non-FIPS-approved primitives both
    /// directions (SHA-1 historical-verify excepted).
    pub fn fips_strict() -> Self {
        Self::builder(PolicyMode::FipsStrict).build()
    }

    /// Start building a policy from `mode`.
    pub fn builder(mode: PolicyMode) -> SecurityPolicyBuilder {
        SecurityPolicyBuilder {
            mode,
            overrides: BTreeMap::new(),
            unknown_algorithm: Decision::Deny,
            min_security_bits: mode.default_min_bits(),
        }
    }

    /// The policy's base mode.
    pub fn mode(&self) -> PolicyMode {
        self.mode
    }

    /// The `min_security_bits` Write floor in effect.
    pub fn min_security_bits(&self) -> u16 {
        self.min_security_bits
    }

    /// Decide whether `alg` may be used for `use_`.
    ///
    /// Pure, allocation-free, and total (every input yields a
    /// `Decision`; fail-closed on any ambiguity).
    pub fn evaluate(&self, alg: AlgorithmId, use_: AlgorithmUse) -> Decision {
        // 1. Explicit overrides.
        if let Some(&d) = self.overrides.get(&(alg, use_)) {
            match d {
                // A deny override is always terminal.
                Decision::Deny => return Decision::Deny,
                // FipsStrict refuses to let an allow-override
                // re-enable a non-approved primitive.
                Decision::Allow => {
                    if self.mode == PolicyMode::FipsStrict && !alg.is_fips_approved() {
                        // fall through to mode default (will Deny)
                    } else {
                        return Decision::Allow;
                    }
                },
            }
        }

        // 2. Mode default matrix.
        match self.mode.default_decision(alg, use_) {
            Decision::Deny => Decision::Deny,
            // 3. min_security_bits Write floor (defence in depth).
            Decision::Allow => {
                if use_ == AlgorithmUse::Write && alg.min_security_bits() < self.min_security_bits {
                    Decision::Deny
                } else {
                    Decision::Allow
                }
            },
        }
    }

    /// Convenience: `evaluate(..) == Decision::Allow`.
    pub fn allows(&self, alg: AlgorithmId, use_: AlgorithmUse) -> bool {
        self.evaluate(alg, use_) == Decision::Allow
    }

    /// The fail-closed decision for an algorithm id this build does
    /// not recognise (always `Deny` in v0.3.50).
    pub fn unknown_algorithm_decision(&self) -> Decision {
        self.unknown_algorithm
    }

    /// Evaluate by string token (the shape the language bindings pass).
    /// An unrecognised algorithm token resolves to
    /// [`Self::unknown_algorithm_decision`] — **fail-closed**, never a
    /// silent allow.
    pub fn evaluate_token(&self, alg_token: &str, use_: AlgorithmUse) -> Decision {
        match AlgorithmId::from_token(alg_token) {
            Some(a) => self.evaluate(a, use_),
            None => self.unknown_algorithm,
        }
    }
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self::compat()
    }
}

/// Builder for [`SecurityPolicy`]. `#[non_exhaustive]`-friendly: new
/// knobs are added as methods without breaking callers.
#[derive(Debug, Clone)]
pub struct SecurityPolicyBuilder {
    mode: PolicyMode,
    overrides: BTreeMap<(AlgorithmId, AlgorithmUse), Decision>,
    unknown_algorithm: Decision,
    min_security_bits: u16,
}

impl SecurityPolicyBuilder {
    /// Explicitly allow `(alg, use_)` (subject to the FipsStrict
    /// non-approved exception — see [`SecurityPolicy`] precedence).
    pub fn allow(mut self, alg: AlgorithmId, use_: AlgorithmUse) -> Self {
        self.overrides.insert((alg, use_), Decision::Allow);
        self
    }

    /// Explicitly deny `(alg, use_)` (terminal — deny always wins).
    pub fn deny(mut self, alg: AlgorithmId, use_: AlgorithmUse) -> Self {
        self.overrides.insert((alg, use_), Decision::Deny);
        self
    }

    /// Set the `min_security_bits` Write floor (e.g. `256` to force
    /// AES-256-only writes under `Strict`).
    pub fn min_security_bits(mut self, bits: u16) -> Self {
        self.min_security_bits = bits;
        self
    }

    /// Decision for an algorithm id this build does not recognise.
    /// Defaults to `Deny` (fail-closed) and v0.3.50 callers should not
    /// relax it; exposed so the forward-shaped API can do so
    /// deliberately later.
    pub fn unknown_algorithm(mut self, decision: Decision) -> Self {
        self.unknown_algorithm = decision;
        self
    }

    /// Finalize the policy.
    pub fn build(self) -> SecurityPolicy {
        SecurityPolicy {
            mode: self.mode,
            overrides: self.overrides,
            unknown_algorithm: self.unknown_algorithm,
            min_security_bits: self.min_security_bits,
        }
    }
}

/// Grammar: `mode[;clause]*` where `mode ∈ {compat,strict,fips-strict}`
/// and `clause = (allow|deny):<alg-token>@<read|write>`. Whitespace
/// around separators is tolerated. Any unknown token is a fail-closed
/// parse error.
impl FromStr for SecurityPolicy {
    type Err = PolicyParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split(';');
        let mode_tok = parts
            .next()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| PolicyParseError("empty policy spec".to_string()))?;
        let mode = match mode_tok {
            "compat" => PolicyMode::Compat,
            "strict" => PolicyMode::Strict,
            "fips-strict" => PolicyMode::FipsStrict,
            other => {
                return Err(PolicyParseError(format!(
                    "unknown mode '{other}' (expected compat|strict|fips-strict)"
                )))
            },
        };
        let mut b = SecurityPolicy::builder(mode);
        for raw in parts {
            let clause = raw.trim();
            if clause.is_empty() {
                continue;
            }
            let (verb, rest) = clause.split_once(':').ok_or_else(|| {
                PolicyParseError(format!("clause '{clause}' must be '<allow|deny>:<alg>@<use>'"))
            })?;
            let (alg_tok, use_tok) = rest.split_once('@').ok_or_else(|| {
                PolicyParseError(format!("clause '{clause}' missing '@<read|write>'"))
            })?;
            let alg = AlgorithmId::from_token(alg_tok.trim())
                .ok_or_else(|| PolicyParseError(format!("unknown algorithm token '{alg_tok}'")))?;
            let use_ = AlgorithmUse::from_token(use_tok.trim()).ok_or_else(|| {
                PolicyParseError(format!("unknown use token '{use_tok}' (expected read|write)"))
            })?;
            b = match verb.trim() {
                "allow" => b.allow(alg, use_),
                "deny" => b.deny(alg, use_),
                other => {
                    return Err(PolicyParseError(format!(
                        "unknown verb '{other}' (expected allow|deny)"
                    )))
                },
            };
        }
        Ok(b.build())
    }
}

impl std::fmt::Display for SecurityPolicy {
    /// Renders back to the canonical grammar (round-trips with
    /// [`FromStr`]). Overrides are emitted in deterministic
    /// `BTreeMap` order.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.mode.token())?;
        for ((alg, use_), d) in &self.overrides {
            let verb = match d {
                Decision::Allow => "allow",
                Decision::Deny => "deny",
            };
            write!(f, ";{verb}:{}@{}", alg.token(), use_.token())?;
        }
        Ok(())
    }
}

/// One policy evaluation, for the audit seam (governance /
/// CBOM-adjacent observability — see the #230 plan §3.7).
#[derive(Debug, Clone, Copy)]
pub struct AuditEvent {
    /// The primitive evaluated.
    pub algorithm: AlgorithmId,
    /// The direction.
    pub use_: AlgorithmUse,
    /// The outcome.
    pub decision: Decision,
    /// The policy mode in effect.
    pub mode: PolicyMode,
}

/// Receives [`AuditEvent`]s. The default is a no-op; a `log`-backed
/// sink ships; consumers inject SIEM/OpenTelemetry sinks without
/// `pdf_oxide` depending on them (Dependency Inversion).
///
/// A sink panic must never turn a deny into an allow or abort the
/// process — the decorator (later increment) calls sinks defensively.
pub trait AuditSink: Send + Sync {
    /// Record one evaluation.
    fn record(&self, event: &AuditEvent);
}

/// The default sink: drops every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopAuditSink;

impl AuditSink for NoopAuditSink {
    fn record(&self, _event: &AuditEvent) {}
}

/// A sink that logs via the `log` crate — `debug!` for allow,
/// `warn!` for deny. No new dependency (`log` is already pervasive).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogAuditSink;

impl AuditSink for LogAuditSink {
    fn record(&self, e: &AuditEvent) {
        match e.decision {
            Decision::Allow => log::debug!(
                "crypto-policy allow: {} {} (mode={})",
                e.algorithm.token(),
                e.use_.token(),
                e.mode.token()
            ),
            Decision::Deny => log::warn!(
                "crypto-policy DENY: {} {} (mode={})",
                e.algorithm.token(),
                e.use_.token(),
                e.mode.token()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §3.4 mode-default matrix, asserted verbatim --------------

    /// Compat allows every (alg, use_) — behaviour-preserving default.
    #[test]
    fn compat_allows_everything() {
        let p = SecurityPolicy::compat();
        for &a in &AlgorithmId::ALL {
            for u in [AlgorithmUse::Read, AlgorithmUse::Write] {
                assert_eq!(p.evaluate(a, u), Decision::Allow, "compat {a:?} {u:?}");
            }
        }
    }

    /// Strict: every algorithm is readable; only FIPS-approved
    /// primitives are writable.
    #[test]
    fn strict_read_all_write_fips_only() {
        let p = SecurityPolicy::strict();
        for &a in &AlgorithmId::ALL {
            assert_eq!(p.evaluate(a, AlgorithmUse::Read), Decision::Allow, "strict R {a:?}");
            let want = if a.is_fips_approved() {
                Decision::Allow
            } else {
                Decision::Deny
            };
            assert_eq!(p.evaluate(a, AlgorithmUse::Write), want, "strict W {a:?}");
        }
        // Spot-check the spec's named cells.
        assert_eq!(p.evaluate(AlgorithmId::HashMd5, AlgorithmUse::Write), Decision::Deny);
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Deny);
        assert_eq!(
            p.evaluate(AlgorithmId::SigRsaPkcs1v15Sha1, AlgorithmUse::Write),
            Decision::Deny
        );
        assert_eq!(p.evaluate(AlgorithmId::CipherAes256Cbc, AlgorithmUse::Write), Decision::Allow);
        assert_eq!(
            p.evaluate(AlgorithmId::SigEcdsaP256Sha256, AlgorithmUse::Write),
            Decision::Allow
        );
    }

    /// FipsStrict: non-approved denied both directions, except SHA-1
    /// *hash* read (historical-signature verification, SP 800-131A).
    #[test]
    fn fips_strict_matrix() {
        let p = SecurityPolicy::fips_strict();
        for &a in &AlgorithmId::ALL {
            if a.is_fips_approved() {
                assert_eq!(p.evaluate(a, AlgorithmUse::Read), Decision::Allow, "fips R {a:?}");
                assert_eq!(p.evaluate(a, AlgorithmUse::Write), Decision::Allow, "fips W {a:?}");
            } else {
                assert_eq!(p.evaluate(a, AlgorithmUse::Write), Decision::Deny, "fips W {a:?}");
                let want_read = if a == AlgorithmId::HashSha1 {
                    Decision::Allow
                } else {
                    Decision::Deny
                };
                assert_eq!(p.evaluate(a, AlgorithmUse::Read), want_read, "fips R {a:?}");
            }
        }
    }

    // ---- decision precedence --------------------------------------

    #[test]
    fn deny_override_is_terminal_even_over_compat_allow() {
        let p = SecurityPolicy::builder(PolicyMode::Compat)
            .deny(AlgorithmId::CipherRc4, AlgorithmUse::Write)
            .build();
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Deny);
        // Read still allowed (override is per-(alg,use)).
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Read), Decision::Allow);
    }

    #[test]
    fn allow_override_beats_strict_write_deny() {
        let p = SecurityPolicy::builder(PolicyMode::Strict)
            .allow(AlgorithmId::CipherRc4, AlgorithmUse::Write)
            .build();
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Allow);
    }

    #[test]
    fn fips_strict_ignores_allow_override_for_non_approved() {
        // The .NET AllowOnlyFipsAlgorithms precedent: you cannot
        // re-enable RC4 under FipsStrict with an allow override.
        let p = SecurityPolicy::builder(PolicyMode::FipsStrict)
            .allow(AlgorithmId::CipherRc4, AlgorithmUse::Read)
            .allow(AlgorithmId::CipherRc4, AlgorithmUse::Write)
            .build();
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Read), Decision::Deny);
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Deny);
        // But an allow override of an approved alg is honoured.
        let p2 = SecurityPolicy::builder(PolicyMode::FipsStrict)
            .allow(AlgorithmId::CipherAes256Cbc, AlgorithmUse::Write)
            .build();
        assert_eq!(p2.evaluate(AlgorithmId::CipherAes256Cbc, AlgorithmUse::Write), Decision::Allow);
    }

    // ---- min_security_bits Write floor ----------------------------

    #[test]
    fn strict_min_bits_256_forbids_aes128_write_allows_read() {
        let p = SecurityPolicy::builder(PolicyMode::Strict)
            .min_security_bits(256)
            .build();
        // AES-128 is FIPS-approved but below the 256-bit Write floor.
        assert_eq!(p.evaluate(AlgorithmId::CipherAes128Cbc, AlgorithmUse::Write), Decision::Deny);
        assert_eq!(p.evaluate(AlgorithmId::CipherAes128Cbc, AlgorithmUse::Read), Decision::Allow);
        assert_eq!(p.evaluate(AlgorithmId::CipherAes256Cbc, AlgorithmUse::Write), Decision::Allow);
        // Default Strict floor (128) allows AES-128 write.
        let d = SecurityPolicy::strict();
        assert_eq!(d.evaluate(AlgorithmId::CipherAes128Cbc, AlgorithmUse::Write), Decision::Allow);
    }

    // ---- grammar round-trip & fail-closed parsing -----------------

    #[test]
    fn parse_modes() {
        assert_eq!("compat".parse::<SecurityPolicy>().unwrap().mode(), PolicyMode::Compat);
        assert_eq!("strict".parse::<SecurityPolicy>().unwrap().mode(), PolicyMode::Strict);
        assert_eq!("fips-strict".parse::<SecurityPolicy>().unwrap().mode(), PolicyMode::FipsStrict);
    }

    #[test]
    fn parse_with_overrides_and_roundtrip() {
        let spec = "compat;deny:rc4@write;deny:md5@write";
        let p: SecurityPolicy = spec.parse().unwrap();
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Write), Decision::Deny);
        assert_eq!(p.evaluate(AlgorithmId::HashMd5, AlgorithmUse::Write), Decision::Deny);
        assert_eq!(p.evaluate(AlgorithmId::CipherRc4, AlgorithmUse::Read), Decision::Allow);
        // Display round-trips (BTreeMap order is deterministic).
        let rendered = p.to_string();
        let reparsed: SecurityPolicy = rendered.parse().unwrap();
        assert_eq!(reparsed.to_string(), rendered);
    }

    #[test]
    fn parse_tolerates_whitespace() {
        let p: SecurityPolicy = " strict ; deny: sha1 @ write ".parse().unwrap();
        assert_eq!(p.mode(), PolicyMode::Strict);
        assert_eq!(p.evaluate(AlgorithmId::HashSha1, AlgorithmUse::Write), Decision::Deny);
    }

    #[test]
    fn parse_fails_closed_on_garbage() {
        for bad in [
            "",
            "garbage-mode",
            "compat;rc4@write",         // missing verb
            "compat;deny:rc4",          // missing @use
            "compat;deny:nope@write",   // unknown alg
            "compat;deny:rc4@sideways", // unknown use
            "compat;maybe:rc4@write",   // unknown verb
        ] {
            assert!(
                bad.parse::<SecurityPolicy>().is_err(),
                "spec {bad:?} must be a fail-closed parse error"
            );
        }
    }

    // ---- audit seam ------------------------------------------------

    #[test]
    fn log_audit_sink_does_not_panic() {
        let s = LogAuditSink;
        s.record(&AuditEvent {
            algorithm: AlgorithmId::CipherRc4,
            use_: AlgorithmUse::Write,
            decision: Decision::Deny,
            mode: PolicyMode::Strict,
        });
        NoopAuditSink.record(&AuditEvent {
            algorithm: AlgorithmId::HashSha256,
            use_: AlgorithmUse::Read,
            decision: Decision::Allow,
            mode: PolicyMode::Compat,
        });
    }

    #[test]
    fn token_roundtrips_for_every_algorithm() {
        for &a in &AlgorithmId::ALL {
            assert_eq!(AlgorithmId::from_token(a.token()), Some(a), "token roundtrip {a:?}");
        }
        assert_eq!(AlgorithmId::from_token("does-not-exist"), None);
    }

    #[test]
    fn kind_and_fips_classification_are_consistent() {
        assert_eq!(AlgorithmId::HashMd5.kind(), AlgorithmKind::Hash);
        assert_eq!(AlgorithmId::CipherRc4.kind(), AlgorithmKind::SymmetricCipher);
        assert_eq!(AlgorithmId::SigRsaPssSha256.kind(), AlgorithmKind::SignatureSign);
        assert!(!AlgorithmId::HashMd5.is_fips_approved());
        assert!(!AlgorithmId::HashSha1.is_fips_approved());
        assert!(!AlgorithmId::CipherRc4.is_fips_approved());
        assert!(!AlgorithmId::SigRsaPkcs1v15Sha1.is_fips_approved());
        assert!(AlgorithmId::HashSha256.is_fips_approved());
        assert!(AlgorithmId::CipherAes256Cbc.is_fips_approved());
        assert!(AlgorithmId::SigEcdsaP384Sha384.is_fips_approved());
    }

    #[test]
    fn default_policy_is_compat() {
        assert_eq!(SecurityPolicy::default().mode(), PolicyMode::Compat);
    }

    #[test]
    fn evaluate_token_fails_closed_on_unknown() {
        let p = SecurityPolicy::compat();
        // Known token follows the normal matrix.
        assert_eq!(p.evaluate_token("aes256", AlgorithmUse::Write), Decision::Allow);
        // Unknown token → fail-closed Deny even under Compat.
        assert_eq!(p.evaluate_token("kyber768", AlgorithmUse::Read), Decision::Deny);
        assert_eq!(p.unknown_algorithm_decision(), Decision::Deny);
    }

    #[test]
    fn unknown_algorithm_decision_is_configurable() {
        let p = SecurityPolicy::builder(PolicyMode::Compat)
            .unknown_algorithm(Decision::Allow)
            .build();
        assert_eq!(p.evaluate_token("future-pqc", AlgorithmUse::Read), Decision::Allow);
    }
}
