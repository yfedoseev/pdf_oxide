<?php

declare(strict_types=1);

namespace PdfOxide\Enums;

/**
 * PAdES (PDF Advanced Electronic Signatures, ETSI EN 319 142-1)
 * conformance level — passed across the FFI boundary as `int32_t level`.
 *
 * Integer ordinals are FROZEN by the Rust ABI
 * (`PadesLevel` in `src/signing/pades.rs`): never renumber. See
 * the v0.3.51 release-plans note ("the PadesLevel lesson"):
 * a frozen wire format is the cheapest cross-binding contract.
 */
enum PadesLevel: int
{
    /** B-B: baseline (CMS signature with required signing attributes). */
    case BB = 0;

    /** B-T: B-B + signed timestamp from a TSA (RFC 3161). */
    case BT = 1;

    /**
     * B-LT: B-T + long-term validation material (DSS: certs, CRLs,
     * OCSPs) embedded for offline verification.
     */
    case BLT = 2;

    /** B-LTA: B-LT + document timestamp on the full /DSS. */
    case BLTA = 3;

    /**
     * Whether this level requires a TSA URL at signing time.
     * (B-B does not; B-T / B-LT / B-LTA all do.)
     */
    public function requiresTsa(): bool
    {
        return $this !== self::BB;
    }
}
