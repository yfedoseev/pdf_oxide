<?php

declare(strict_types=1);

/**
 * PHPUnit Bootstrap.
 *
 * Loads the autoloader (composer-managed or hand-rolled PSR-4
 * fallback) and exports two test-shared constants:
 *
 *   PDF_OXIDE_NATIVE_LIB  — absolute path to the cdylib, or null.
 *                            Resolution order:
 *                              1. `PDF_OXIDE_CDYLIB_PATH` env var
 *                              2. `target/release/libpdf_oxide.{so,dylib,dll}`
 *                                 relative to the repo root
 *                              3. system locations (`/usr/local/lib/`)
 *   PDF_OXIDE_SAMPLE_PDF  — absolute path to the smallest test PDF
 *                            available (preferring `php/tests/fixtures/`
 *                            then `tests/fixtures/simple.pdf` at the
 *                            repo root).
 *
 * The Integration test suite self-skips whenever PDF_OXIDE_NATIVE_LIB
 * is null, keeping the Unit suite runnable on any box without the
 * cdylib being present (CI uploads them as separate jobs).
 */

// ---------- Autoloader ----------
$autoloader = dirname(__DIR__) . '/vendor/autoload.php';
if (file_exists($autoloader)) {
    require $autoloader;
} else {
    // PSR-4 fallback so tests can run without `composer install`
    // (CI does install; local dev sometimes doesn't).
    spl_autoload_register(function (string $class): void {
        $prefix = 'PdfOxide\\';
        if (! str_starts_with($class, $prefix)) {
            return;
        }
        $relative = substr($class, strlen($prefix));
        $base = dirname(__DIR__) . '/src/';
        $path = $base . str_replace('\\', '/', $relative) . '.php';
        if (is_file($path)) {
            require $path;
        }
    });
    // Tests namespace fallback (PdfOxide\Tests\…) — only needed when
    // composer hasn't generated the dev-autoload map.
    spl_autoload_register(function (string $class): void {
        $prefix = 'PdfOxide\\Tests\\';
        if (! str_starts_with($class, $prefix)) {
            return;
        }
        $relative = substr($class, strlen($prefix));
        $base = __DIR__ . '/';
        $path = $base . str_replace('\\', '/', $relative) . '.php';
        if (is_file($path)) {
            require $path;
        }
    });
}

error_reporting(E_ALL);
ini_set('display_errors', '1');

// ---------- Native library resolution ----------
$repoRoot = dirname(__DIR__, 2);

$nativeLib = null;
$envOverride = getenv('PDF_OXIDE_CDYLIB_PATH');
if (is_string($envOverride) && $envOverride !== '' && is_file($envOverride)) {
    $nativeLib = $envOverride;
}
if ($nativeLib === null) {
    foreach (
        [
            $repoRoot . '/target/release/libpdf_oxide.so',
            $repoRoot . '/target/release/libpdf_oxide.dylib',
            $repoRoot . '/target/release/pdf_oxide.dll',
            $repoRoot . '/target/release/libpdf_oxide.dll',
            '/usr/local/lib/libpdf_oxide.so',
            '/usr/local/lib/libpdf_oxide.dylib',
        ] as $candidate
    ) {
        if (is_file($candidate)) {
            $nativeLib = $candidate;
            break;
        }
    }
}
define('PDF_OXIDE_NATIVE_LIB', $nativeLib);

// ---------- Fixture resolution ----------
$localFixtures = __DIR__ . '/fixtures';
if (! is_dir($localFixtures)) {
    @mkdir($localFixtures, 0o777, true);
}
define('TEST_FIXTURES_DIR', $localFixtures);

$samplePdf = null;
// Prefer the binding's own fixtures (committed under php/tests/fixtures/);
// fall back to the upstream repo root.
$candidates = [
    $localFixtures . '/tiny.pdf',
    $localFixtures . '/simple.pdf',
    $repoRoot . '/tests/fixtures/simple.pdf',
    $repoRoot . '/tests/fixtures/1.pdf',
];
foreach ($candidates as $candidate) {
    if (is_file($candidate)) {
        $samplePdf = $candidate;
        break;
    }
}
define('PDF_OXIDE_SAMPLE_PDF', $samplePdf);
