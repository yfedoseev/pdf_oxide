<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\PdfDocument;
use PHPUnit\Framework\TestCase;

/**
 * Integration tests for OutlineManager — both the read-side
 * (`getCount` / `getAll`, exercising the Phase 7 scaffold-bug fix)
 * and the v0.3.50 #482 split-by-bookmarks planner.
 *
 * @requires extension ffi
 */
final class OutlineSplitTest extends TestCase
{
    protected function setUp(): void
    {
        if (! extension_loaded('ffi')) {
            $this->markTestSkipped('ext-ffi not loaded');
        }
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide cdylib not found.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testNoOutlinesDocReturnsZeroCount(): void
    {
        // Pre-Phase 7 scaffold bug: `getCount()` called a phantom C
        // symbol (`pdf_document_get_outline_count`). The Phase 7 fix
        // routes through `pdf_document_get_outline()` which returns
        // `[]` for outline-less PDFs.
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $outlines = $doc->outlines();

        $count = $outlines->getCount();
        $this->assertIsInt($count);
        $this->assertGreaterThanOrEqual(0, $count);
        $this->assertFalse($outlines->hasOutlines(), 'tiny.pdf has no outlines.');
        $this->assertSame([], $outlines->getAll());
    }

    public function testPlanSplitReturnsArray(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $plan = $doc->outlines()->planSplit();
        $this->assertIsArray($plan, 'planSplit should always return an array (possibly empty).');
    }

    public function testPlanSplitAcceptsOptions(): void
    {
        $doc = new PdfDocument(PDF_OXIDE_SAMPLE_PDF);
        $plan = $doc->outlines()->planSplit(['min_level' => 1, 'max_level' => 2]);
        $this->assertIsArray($plan);
    }
}
