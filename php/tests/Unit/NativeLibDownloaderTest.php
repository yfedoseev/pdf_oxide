<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit;

use PHPUnit\Framework\TestCase;

/**
 * Smoke test for the Composer post-install native-library downloader
 * helper functions in `scripts/download-native-lib.php`.
 *
 * Loads the script with `require` (the entry-point check at the bottom
 * is `realpath`-guarded so this is safe) and exercises the platform
 * detection logic without actually downloading anything.
 */
final class NativeLibDownloaderTest extends TestCase
{
    protected function setUp(): void
    {
        $path = dirname(__DIR__, 2) . '/scripts/download-native-lib.php';
        if (! is_file($path)) {
            $this->markTestSkipped('download-native-lib.php not present.');
        }
        require_once $path;
    }

    public function testDetectPlatformReturnsKnownKey(): void
    {
        $platform = \detectPlatform();
        $this->assertIsArray($platform);
        $this->assertArrayHasKey('key', $platform);
        $this->assertArrayHasKey('lib_name', $platform);
        $this->assertContains($platform['key'], [
            'linux-x86_64', 'linux-aarch64',
            'darwin-x86_64', 'darwin-arm64',
            'windows-x64',
        ], "detectPlatform should return a known platform key, got {$platform['key']}");
    }

    public function testLibraryNameMatchesPlatform(): void
    {
        $platform = \detectPlatform();
        $expected = match (true) {
            str_starts_with($platform['key'], 'linux-') => 'libpdf_oxide.so',
            str_starts_with($platform['key'], 'darwin-') => 'libpdf_oxide.dylib',
            str_starts_with($platform['key'], 'windows-') => 'pdf_oxide.dll',
            default => null,
        };
        $this->assertSame($expected, $platform['lib_name']);
    }
}
