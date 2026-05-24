<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfPolicy;

/**
 * Smoke tests for {@see PdfPolicy}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/PdfPolicyTest.java`.
 *
 * pdf_oxide is set-once per process — we DON'T call `set()` here so
 * we don't lock the policy for the rest of the PHPUnit run. Other
 * integration tests open documents (which may lazily seed the policy
 * to `compat`), so we only test read-only accessors + preset returns.
 */
final class PdfPolicyTest extends IntegrationTestCase
{
    public function testCurrentReturnsString(): void
    {
        $current = PdfPolicy::current();
        $this->assertIsString($current);
        $this->assertNotEmpty($current);
    }

    public function testPresetAccessors(): void
    {
        $this->assertSame('compat', PdfPolicy::compat());
        $this->assertSame('strict', PdfPolicy::strict());
        $this->assertSame('fips_strict', PdfPolicy::fipsStrict());
    }

    public function testFipsAvailableReturnsBool(): void
    {
        $this->assertIsBool(PdfPolicy::fipsAvailable());
    }

    public function testActiveProviderReturnsString(): void
    {
        $this->assertIsString(PdfPolicy::activeProvider());
    }
}
