<?php

declare(strict_types=1);

/**
 * PHPUnit Bootstrap File.
 *
 * Loads the autoloader and sets up the test environment for both unit
 * and integration tests. Integration tests need the cdylib + a sample
 * PDF; this file locates both and exposes them via constants.
 *
 * Behaviour when the cdylib is absent:
 *   - The constant PDF_OXIDE_NATIVE_LIB is set to `null`.
 *   - Integration tests must read that constant and `markTestSkipped()`
 *     when it's null — keeping unit tests runnable on any box.
 */

// Load autoloader
$autoloader = dirname(__DIR__) . '/vendor/autoload.php';
if (file_exists($autoloader)) {
    require $autoloader;
} else {
    // Fall back to a minimal PSR-4 autoloader so smoke tests can load
    // src/ without composer. Useful in dev when `composer install`
    // hasn't been run yet.
    spl_autoload_register(function (string $class): void {
        $prefix = 'PdfOxide\\';
        if (! str_starts_with($class, $prefix)) {
            return;
        }
        $relative = substr($class, strlen($prefix));
        $path = dirname(__DIR__) . '/src/' . str_replace('\\', '/', $relative) . '.php';
        if (is_file($path)) {
            require $path;
        }
    });
}

// Standard test environment.
error_reporting(E_ALL);
ini_set('display_errors', '1');

// Fixtures directory (we reuse the upstream Rust test fixtures so we
// don't ship duplicate PDFs in the package).
$repoRoot = dirname(__DIR__, 2);
$upstreamFixtures = $repoRoot . '/tests/fixtures';
$localFixtures = __DIR__ . '/Fixtures';
if (! is_dir($localFixtures)) {
    @mkdir($localFixtures, 0777, true);
}

// Pick a representative small PDF for the smoke tests.
$samplePdf = null;
foreach (['1.pdf'] as $candidate) {
    $path = $upstreamFixtures . '/' . $candidate;
    if (is_file($path)) {
        $samplePdf = $path;
        break;
    }
}
if ($samplePdf === null) {
    // Last-ditch: look in tests/Fixtures itself.
    $candidates = glob($localFixtures . '/*.pdf') ?: [];
    if (! empty($candidates)) {
        $samplePdf = $candidates[0];
    }
}
define('PDF_OXIDE_SAMPLE_PDF', $samplePdf);
define('TEST_FIXTURES_DIR', $localFixtures);

// Locate the native library (best-effort; integration tests should
// skip themselves if it's null).
$nativeLib = null;
foreach (
    [
        $repoRoot . '/target/release/libpdf_oxide.so',
        $repoRoot . '/target/release/libpdf_oxide.dylib',
        $repoRoot . '/target/release/pdf_oxide.dll',
        '/usr/local/lib/libpdf_oxide.so',
        '/usr/local/lib/libpdf_oxide.dylib',
    ] as $candidate
) {
    if (is_file($candidate)) {
        $nativeLib = $candidate;
        break;
    }
}
define('PDF_OXIDE_NATIVE_LIB', $nativeLib);
