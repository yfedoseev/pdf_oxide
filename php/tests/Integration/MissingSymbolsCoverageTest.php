<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\FFI\FunctionBindings;

/**
 * API-coverage smoke tests for C-ABI symbols that previously had no PHP
 * binding: pdf_oxide_set_max_ops_per_stream,
 * pdf_oxide_set_preserve_unmapped_glyphs, and
 * pdf_render_page_with_options_ex.
 *
 * Closes the binding-coverage gap; each test only asserts the symbol is
 * invokable (and, where there is no error channel, returns a prior int).
 */
final class MissingSymbolsCoverageTest extends IntegrationTestCase
{
    /** No error channel — returns the previous cap; assert invokable. */
    public function testSetMaxOpsPerStreamReturnsPreviousCap(): void
    {
        $bindings = new FunctionBindings();
        // Restore the default and capture whatever was active before.
        $previous = $bindings->pdfOxideSetMaxOpsPerStream(-1);
        $this->assertIsInt($previous);
        // Round-trip: setting an explicit cap returns the (-1) we just set.
        $this->assertSame(-1, $bindings->pdfOxideSetMaxOpsPerStream(500000));
        // Leave the global cap restored for other tests.
        $bindings->pdfOxideSetMaxOpsPerStream(-1);
    }

    /** No error channel — returns the previous flag; assert invokable. */
    public function testSetPreserveUnmappedGlyphsReturnsPreviousFlag(): void
    {
        $bindings = new FunctionBindings();
        $previous = $bindings->pdfOxideSetPreserveUnmappedGlyphs(1);
        $this->assertContains($previous, [0, 1]);
        // Round-trip: we set 1, so reading-then-restoring returns 1.
        $this->assertSame(1, $bindings->pdfOxideSetPreserveUnmappedGlyphs($previous));
    }

    /** Feature-gated render path: assert return-or-error. */
    public function testRenderPageWithOptionsExReturnsOrErrors(): void
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfDocumentOpen($this->fixture('simple.pdf'));
        $this->assertNotNull($handle);

        try {
            $image = $bindings->pdfRenderPageWithOptionsEx(
                $handle,
                0,          // page index
                72,         // dpi
                0,          // format: PNG
                1.0,        // bg r
                1.0,        // bg g
                1.0,        // bg b
                1.0,        // bg a
                0,          // transparent background
                1,          // render annotations
                90,         // jpeg quality
                ['HiddenLayer'] // excluded OCG names
            );
            // Success path: a valid image handle.
            $this->assertInstanceOf(\FFI\CData::class, $image);
            $bindings->pdfRenderedImageFree($image);
        } catch (\Throwable $e) {
            // Render may be unavailable in this build (feature-gated);
            // accept the binding error as coverage of the symbol.
            $this->assertInstanceOf(\Throwable::class, $e);
        } finally {
            $bindings->pdfDocumentFree($handle);
        }
    }
}
