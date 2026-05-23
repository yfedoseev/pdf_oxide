<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Managers\WatermarkManager;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for the WatermarkManager class.
 *
 * The watermark FFI surface operates on a page-builder (a Pdf::create
 * builder handle), which the v0.3.55 PHP creation API only partially
 * exposes (Phase 6.10 is deferred to v0.3.56). For this Phase 6 smoke
 * test we exercise the class loader + reflection of the public method
 * surface; full builder-flow tests will land alongside the v0.3.56
 * creation API.
 */
final class WatermarkManagerTest extends TestCase
{
    public function testClassLoadsAndExposesExpectedMethods(): void
    {
        $rc = new \ReflectionClass(WatermarkManager::class);
        $this->assertTrue($rc->hasMethod('addText'));
        $this->assertTrue($rc->hasMethod('addConfidential'));
        $this->assertTrue($rc->hasMethod('addDraft'));
        $this->assertTrue($rc->hasMethod('addStamp'));
        $this->assertTrue($rc->hasMethod('addFreetext'));
    }
}
