<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PHPUnit\Framework\TestCase;

/**
 * Shared base for integration tests. Self-skips the whole class when
 * the cdylib isn't reachable (the contract per the bootstrap.php
 * comment block).
 */
abstract class IntegrationTestCase extends TestCase
{
    protected function setUp(): void
    {
        if (! defined('PDF_OXIDE_NATIVE_LIB') || PDF_OXIDE_NATIVE_LIB === null) {
            $this->markTestSkipped('pdf_oxide cdylib not available — skipping integration test');
        }
    }

    /**
     * Resolve a fixture from the repo-wide `tests/fixtures/` (preferred)
     * or the binding-local `php/tests/fixtures/`. Skips the test if the
     * fixture is missing.
     */
    protected function fixture(string $name): string
    {
        $candidates = [
            // Local fixtures dir (php/tests/fixtures/).
            __DIR__ . '/../fixtures/' . $name,
            // Repo-wide fixtures (tests/fixtures/).
            __DIR__ . '/../../../tests/fixtures/' . $name,
        ];
        foreach ($candidates as $c) {
            if (is_file($c)) {
                return (string) realpath($c);
            }
        }
        $this->markTestSkipped("fixture not found: {$name}");
    }
}
