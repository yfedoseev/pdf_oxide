<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Managers\RedactionManager;
use PdfOxide\Types\Rect;
use PHPUnit\Framework\TestCase;

/**
 * Smoke test for v0.3.50 #231 destructive redaction.
 *
 * The redaction ABI operates on a `DocumentEditor*` (not on the
 * read-only `PdfDocument*` returned by `PdfDocument::open()`); the
 * manager exposes a {@see RedactionManager::openFile()} factory that
 * wires this up correctly.
 */
final class RedactionManagerTest extends TestCase
{
    protected function setUp(): void
    {
        if (PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('libpdf_oxide not built; run `cargo build --release` first.');
        }
        if (PDF_OXIDE_SAMPLE_PDF === null) {
            $this->markTestSkipped('No sample PDF available.');
        }
    }

    public function testOpenFileAndPendingCount(): void
    {
        $redact = RedactionManager::openFile(PDF_OXIDE_SAMPLE_PDF);
        $this->assertInstanceOf(RedactionManager::class, $redact);

        $count = $redact->pendingCount(0);
        $this->assertIsInt($count);
        $this->assertGreaterThanOrEqual(0, $count);
    }

    public function testMarkIncrementsPendingCount(): void
    {
        $redact = RedactionManager::openFile(PDF_OXIDE_SAMPLE_PDF);
        $before = $redact->pendingCount(0);

        $redact->mark(0, new Rect(100, 100, 50, 20));

        $after = $redact->pendingCount(0);
        $this->assertSame($before + 1, $after, 'mark() should add one pending redaction.');
    }
}
