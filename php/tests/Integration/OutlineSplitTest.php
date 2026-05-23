<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for v0.3.50 #482 split-by-bookmarks planning.
 *
 * The sample PDF likely doesn't have outlines; planSplit() should
 * return an empty array gracefully rather than throw.
 */
final class OutlineSplitTest extends TestCase
{
    protected function setUp(): void
    {
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide not built.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testPlanSplitReturnsArray(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $outlines = $doc->outlines();
        $plan = $outlines->planSplit();
        $this->assertIsArray($plan, 'planSplit should always return an array (possibly empty).');
    }

    public function testPlanSplitAcceptsOptions(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $outlines = $doc->outlines();
        $plan = $outlines->planSplit(['min_level' => 1, 'max_level' => 2]);
        $this->assertIsArray($plan);
    }
}
