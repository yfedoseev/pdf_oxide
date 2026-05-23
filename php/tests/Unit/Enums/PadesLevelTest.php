<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Enums;

use PdfOxide\Enums\PadesLevel;
use PHPUnit\Framework\TestCase;

/**
 * Unit tests for {@see PadesLevel}.
 *
 * Integer ordinals are FROZEN by the Rust ABI (`PadesLevel` in
 * `src/signing/pades.rs`). Renumbering would silently corrupt
 * signatures across binding boundaries — guarded here.
 */
final class PadesLevelTest extends TestCase
{
    public function testOrdinalsFrozen(): void
    {
        $this->assertSame(0, PadesLevel::BB->value);
        $this->assertSame(1, PadesLevel::BT->value);
        $this->assertSame(2, PadesLevel::BLT->value);
        $this->assertSame(3, PadesLevel::BLTA->value);
    }

    public function testRequiresTsaContract(): void
    {
        $this->assertFalse(PadesLevel::BB->requiresTsa(), 'B-B does not need a TSA.');
        $this->assertTrue(PadesLevel::BT->requiresTsa(), 'B-T requires a TSA timestamp.');
        $this->assertTrue(PadesLevel::BLT->requiresTsa(), 'B-LT requires a TSA timestamp.');
        $this->assertTrue(PadesLevel::BLTA->requiresTsa(), 'B-LTA requires a TSA timestamp.');
    }

    public function testTryFromAcceptsIntegerOrdinal(): void
    {
        $this->assertSame(PadesLevel::BB, PadesLevel::tryFrom(0));
        $this->assertSame(PadesLevel::BT, PadesLevel::tryFrom(1));
        $this->assertSame(PadesLevel::BLT, PadesLevel::tryFrom(2));
        $this->assertSame(PadesLevel::BLTA, PadesLevel::tryFrom(3));
        $this->assertNull(PadesLevel::tryFrom(99));
    }
}
