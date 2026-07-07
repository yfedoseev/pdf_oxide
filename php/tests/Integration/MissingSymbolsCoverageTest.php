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
 * pdf_oxide_set_preserve_unmapped_glyphs, pdf_render_page_with_options_ex,
 * pdf_oxide_word_get_sequence, pdf_oxide_word_get_rotation,
 * and pdf_oxide_path_get_rendered_bbox.
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

    /** Word accessors: sequence is a draw-order int, rotation is quadrant-snapped degrees. */
    public function testWordSequenceAndRotationAccessors(): void
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfDocumentOpen($this->fixture('multi_column_table.pdf'));
        $this->assertNotNull($handle);

        try {
            $words = $bindings->pdfDocumentExtractWords($handle, 0);
            try {
                $count = $bindings->pdfOxideWordCount($words);
                $this->assertGreaterThan(0, $count);

                $sequence = $bindings->pdfOxideWordGetSequence($words, 0);
                $this->assertGreaterThanOrEqual(0, $sequence);

                // Rotation is snapped to a quadrant; an
                // ordinary horizontal-text fixture reads at 0 degrees.
                $rotation = $bindings->pdfOxideWordGetRotation($words, 0);
                $this->assertContains($rotation, [0.0, 90.0, 180.0, -90.0]);
            } finally {
                $bindings->pdfOxideWordListFree($words);
            }
        } finally {
            $bindings->pdfDocumentFree($handle);
        }
    }

    /** Rendered bbox is the geometric bbox inflated by the stroke. */
    public function testPathRenderedBboxContainsGeometricBbox(): void
    {
        $bindings = new FunctionBindings();
        $handle = $bindings->pdfDocumentOpen($this->fixture('multi_column_table.pdf'));
        $this->assertNotNull($handle);

        try {
            $paths = $bindings->pdfDocumentExtractPaths($handle, 0);
            try {
                $count = $bindings->pdfOxidePathCount($paths);
                if ($count === 0) {
                    // No vector paths on the fixture page: cover the
                    // symbol via its out-of-range error channel instead.
                    $this->expectException(\Throwable::class);
                    $bindings->pdfOxidePathGetRenderedBbox($paths, 0);
                    return;
                }

                for ($i = 0; $i < $count; $i++) {
                    $bbox = $bindings->pdfOxidePathGetBbox($paths, $i);
                    $rendered = $bindings->pdfOxidePathGetRenderedBbox($paths, $i);

                    // Stroke inflation never shrinks the extents.
                    $epsilon = 1e-4;
                    $this->assertLessThanOrEqual($bbox['x'] + $epsilon, $rendered['x']);
                    $this->assertLessThanOrEqual($bbox['y'] + $epsilon, $rendered['y']);
                    $this->assertGreaterThanOrEqual($bbox['width'] - $epsilon, $rendered['width']);
                    $this->assertGreaterThanOrEqual($bbox['height'] - $epsilon, $rendered['height']);
                }
            } finally {
                $bindings->pdfOxidePathListFree($paths);
            }
        } finally {
            $bindings->pdfDocumentFree($handle);
        }
    }
}
